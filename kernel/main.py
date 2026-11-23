"""STL Factory : texte ou image -> modele 3D STL repare (Kaggle, GPU T4).

Chaque etape tourne dans son propre processus pour liberer la memoire GPU :
  bootstrap : cree le venv, installe les dependances, enchaine les etapes
  image     : texte -> image de reference (SDXL) ; sautee si une image est fournie
  infer     : image -> maillage (TRELLIS.2, cascade 512 -> 1024) -> STL brut
  repair    : nettoyage CuMesh + controle de derive -> STL repare
"""
import gc
import inspect
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import traceback
import unicodedata

# Ligne remplacee par stl_factory.py au moment du push. Ne pas modifier le marqueur.
JOB = {}  # @@JOB@@

DEFAULTS = dict(
    mode='text',                 # 'text' ou 'image'
    prompt=None,                 # description de l'objet (anglais conseille)
    name=None,                   # nom des fichiers de sortie
    size_mm=100.0,               # plus grande dimension du STL
    seed=42,
    image_model='stabilityai/stable-diffusion-xl-base-1.0',
    image_steps=30,
    guidance=7.0,
    simplify_faces=3_000_000,
)
JOB = {**DEFAULTS, **JOB, **json.loads(os.environ.get('STL_JOB', '{}'))}


def slugify(text):
    text = unicodedata.normalize('NFKD', text or '').encode('ascii', 'ignore').decode()
    text = re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')
    return text[:40].strip('-') or 'objet'


SLUG = slugify(JOB['name'] or JOB['prompt'])
SIZE = float(JOB['size_mm'])
SIZE_TAG = f'{SIZE:g}mm'
RESOLUTION = 1024
REVISION = '75fbf0183001ed9876c8dbb35de6b68552ee08bd'
ENCODER_BYTES = 1212559808
CHUNK = 65536
BACKGROUND_MODEL = 'ZhengPeng7/BiRefNet'   # licence MIT, usage commercial autorise

ROOT = pathlib.Path('/kaggle/temp/stl-factory')
OUT = pathlib.Path('/kaggle/working') / SLUG
OUT.mkdir(parents=True, exist_ok=True)
REFERENCE = OUT / 'reference.png'
RAW_STL = OUT / f'{SLUG}-{SIZE_TAG}.stl'
REPAIRED_STL = OUT / f'{SLUG}-repaired-{SIZE_TAG}.stl'

os.environ.update(ATTN_BACKEND='xformers', SPARSE_ATTN_BACKEND='xformers',
                  CUDA_VISIBLE_DEVICES='0', PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
                  HF_HOME=str(ROOT / 'hf'), PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='1')

# Suffixe et negatif pour obtenir un objet unique, centre, sur fond neutre :
# c'est ce que TRELLIS.2 reconstruit le mieux.
PROMPT_SUFFIX = (', single object, centered, full view, entire object visible, '
                 'isolated on plain white background, soft studio lighting, '
                 'detailed 3D sculpture, matte material')
NEGATIVE = ('multiple objects, cropped, cut off, out of frame, text, watermark, logo, '
            'busy background, scenery, floor, harsh shadows, blurry, low quality, people')


# --------------------------------------------------------------------------- utilitaires

def run(args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def record(status, file='status.json', **kwargs):
    result = dict(status=status, job=JOB, slug=SLUG, resolution=RESOLUTION,
                  source_revision=REVISION, **kwargs)
    (OUT / file).write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, default=str), flush=True)


def encoder_folder():
    candidates = [p.parent for p in pathlib.Path('/kaggle/input').rglob('model.safetensors')
                  if (p.parent / 'config.json').is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f'Attendu un seul encodeur officiel en entree, trouve {len(candidates)}')
    folder = candidates[0]
    if (folder / 'model.safetensors').stat().st_size != ENCODER_BYTES:
        raise RuntimeError("La taille des poids de l'encodeur ne correspond pas au dataset verifie")
    return folder


# --------------------------------------------------------------------------- bootstrap

def bootstrap():
    if JOB['mode'] not in ('text', 'image'):
        raise ValueError(f"mode inconnu : {JOB['mode']}")
    if JOB['mode'] == 'text' and not JOB['prompt']:
        raise ValueError('Mode texte sans prompt')
    encoder_folder()
    if JOB['mode'] == 'image':
        found = list(pathlib.Path('/kaggle/input').rglob('reference.png'))
        if len(found) != 1:
            raise RuntimeError(f'Mode image : attendu un seul reference.png, trouve {found}')
        shutil.copy(found[0], REFERENCE)
    record('installing')

    ROOT.mkdir(parents=True, exist_ok=True)
    run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
    run([sys.executable, '-m', 'uv', 'venv', '--seed', '--python', '3.10', ROOT / 'venv'])
    python = ROOT / 'venv/bin/python'
    run([python, '-m', 'pip', 'install', '-q', 'torch==2.6.0', 'torchvision==0.21.0',
         'xformers==0.0.29.post3', '--index-url', 'https://download.pytorch.org/whl/cu124'])
    run([python, '-m', 'pip', 'install', '-q', 'transformers==4.57.1', 'accelerate',
         'diffusers>=0.30,<0.36', 'pillow', 'opencv-python-headless', 'easydict', 'tqdm',
         'scipy', 'trimesh', 'safetensors', 'huggingface_hub<1', 'einops', 'timm', 'plyfile',
         'zstandard', 'imageio', 'imageio-ffmpeg', 'matplotlib', 'kornia', 'rtree',
         'pymeshfix==0.17.1',
         'git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8'])
    wheels = []
    for prefix in ('cumesh-', 'o_voxel-', 'flex_gemm-', 'nvdiffrast-'):
        matches = list(pathlib.Path('/kaggle/input').rglob(prefix + '*.whl'))
        if len(matches) != 1:
            raise RuntimeError(f'Attendu une seule wheel {prefix}, trouve {matches}')
        wheels.extend(matches)
    run([python, '-m', 'pip', 'install', '--no-deps', *wheels])
    repo = ROOT / 'TRELLIS.2'
    if not repo.exists():
        run(['git', 'clone', 'https://github.com/microsoft/TRELLIS.2.git', repo])
    run(['git', 'checkout', REVISION], cwd=repo)

    me = pathlib.Path(__file__).resolve()
    env = dict(os.environ, PYTHONPATH=str(repo), STL_JOB=json.dumps(JOB))
    started = time.monotonic()
    if JOB['mode'] == 'text':
        run([python, '-u', me], env=dict(env, STL_STAGE='image'), timeout=1800)
    run([python, '-u', me], env=dict(env, STL_STAGE='infer'), timeout=10800)
    try:
        run([python, '-u', me], env=dict(env, STL_STAGE='repair'), timeout=3600)
    except subprocess.SubprocessError as exc:
        (OUT / 'repair_failed.txt').write_text(
            f'{exc}\nLe STL brut reste disponible, NON valide pour impression.')
        print('Reparation echouee ; STL brut conserve.', flush=True)
    files = sorted(p.name for p in OUT.iterdir() if p.is_file())
    record('finished', file='summary.json', files=files,
           total_seconds=round(time.monotonic() - started),
           repaired=REPAIRED_STL.exists())


# --------------------------------------------------------------------------- etape image

def make_image():
    import numpy as np
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    record('generating_image')
    # Le VAE SDXL d'origine produit des NaN (images noires) en fp16 ; celui-ci est corrige.
    vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16)
    kwargs = dict(vae=vae, torch_dtype=torch.float16, use_safetensors=True)
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(JOB['image_model'], variant='fp16', **kwargs)
    except (OSError, ValueError):
        pipe = StableDiffusionXLPipeline.from_pretrained(JOB['image_model'], **kwargs)
    pipe.to('cuda')
    pipe.enable_vae_slicing()
    prompt = JOB['prompt'].strip().rstrip('.') + PROMPT_SUFFIX
    generator = torch.Generator('cuda').manual_seed(int(JOB['seed']))
    image = pipe(prompt=prompt, negative_prompt=NEGATIVE,
                 num_inference_steps=int(JOB['image_steps']), guidance_scale=float(JOB['guidance']),
                 width=1024, height=1024, generator=generator).images[0]
    if np.asarray(image).max() < 10:
        raise RuntimeError("SDXL a produit une image noire (probablement des NaN)")
    image.save(REFERENCE)
    record('image_ready', full_prompt=prompt, image='reference.png')


# --------------------------------------------------------------------------- etape TRELLIS.2

def infer():
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines.base import Pipeline
    from trellis2.pipelines import samplers
    from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

    record('loading_models')
    # La cascade 1024 de Microsoft utilise les modeles de forme 512 et 1024.
    Trellis2ImageTo3DPipeline.model_names_to_load = [
        'sparse_structure_flow_model', 'sparse_structure_decoder',
        'shape_slat_flow_model_512', 'shape_slat_flow_model_1024', 'shape_slat_decoder']
    pipe = Pipeline.from_pretrained.__func__(Trellis2ImageTo3DPipeline, 'microsoft/TRELLIS.2-4B')
    cfg = pipe._pretrained_args
    for stage in ('sparse_structure', 'shape_slat'):
        spec = cfg[stage + '_sampler']
        setattr(pipe, stage + '_sampler', getattr(samplers, spec['name'])(**spec['args']))
        setattr(pipe, stage + '_sampler_params', spec['params'])
    pipe.shape_slat_normalization = cfg['shape_slat_normalization']
    pipe.image_cond_model = DinoV3FeatureExtractor(str(encoder_folder()))
    from trellis2.pipelines.rembg import BiRefNet
    # Detourage : BiRefNet public (licence MIT). Le modele configure par Microsoft,
    # briaai/RMBG-2.0, est sous licence non commerciale : ne pas l'utiliser pour des STL vendus.
    background_model = BACKGROUND_MODEL
    pipe.rembg_model = BiRefNet(background_model)
    pipe.low_vram = True
    pipe._device = torch.device('cuda:0')
    for name, model in pipe.models.items():
        if hasattr(model, 'convert_to'):
            model.convert_to(torch.float32 if name == 'sparse_structure_flow_model' else torch.float16)

    image = pipe.preprocess_image(Image.open(REFERENCE))
    image.save(OUT / 'conditioning.png')
    record('preprocessed', background_model=background_model, conditioning_size=list(image.size))
    pipe.rembg_model = None
    gc.collect()
    torch.cuda.empty_cache()
    torch.manual_seed(int(JOB['seed']))
    started = time.monotonic()
    with torch.inference_mode():
        record('encoding_image')
        cond = pipe.get_cond([image], 512)
        if not torch.isfinite(cond['cond']).all():
            raise RuntimeError('DINO a produit un conditionnement non fini')
        cond_hr = pipe.get_cond([image], 1024)
        if not torch.isfinite(cond_hr['cond']).all():
            raise RuntimeError('DINO a produit un conditionnement 1024 non fini')
        torch.cuda.empty_cache()
        record('sampling_structure')
        coords = pipe.sample_sparse_structure(cond, 32)
        if len(coords) == 0:
            raise RuntimeError('Le sampler de structure a produit zero voxel occupe')
        torch.cuda.empty_cache()
        record('sampling_shape', structure_tokens=len(coords))
        latent, actual_resolution = pipe.sample_shape_slat_cascade(
            cond, cond_hr, pipe.models['shape_slat_flow_model_512'],
            pipe.models['shape_slat_flow_model_1024'], 512, RESOLUTION, coords,
            max_num_tokens=1_000_000_000)
        if actual_resolution != RESOLUTION:
            raise RuntimeError(f'Resolution de generation inattendue : {actual_resolution}')
        torch.cuda.empty_cache()
        record('decoding_shape')
        meshes, _subs = pipe.decode_shape_slat(latent, RESOLUTION)
        vertices = meshes[0].vertices.cpu().numpy()
        faces = meshes[0].faces.cpu().numpy()
    if not np.isfinite(vertices).all() or len(faces) == 0:
        raise RuntimeError('Le decodeur a renvoye un maillage invalide ou vide')
    del latent, cond, cond_hr, coords, meshes, _subs, pipe, model
    gc.collect()
    torch.cuda.empty_cache()
    source = trimesh.Trimesh(vertices, faces, process=False)
    raw_stats = dict(vertices=len(vertices), faces=len(faces),
                     watertight=bool(source.is_watertight),
                     winding_consistent=bool(source.is_winding_consistent))
    del source
    record('remeshing', raw=raw_stats)

    # Partie geometrie de l'exporteur Microsoft, avec un remeshing par blocs pour le T4.
    import cumesh
    module = cumesh.remeshing
    module._sample_distances = sample_distances
    module._chunk_topology = chunk_topology
    bounded_source = patched_source(inspect.getsource(module))
    exec(compile(bounded_source, '<chunked-native-UDF>', 'exec'), module.__dict__)
    gpu_mesh = cumesh.CuMesh()
    gpu_mesh.init(torch.as_tensor(vertices, device='cuda', dtype=torch.float32).contiguous(),
                  torch.as_tensor(faces, device='cuda', dtype=torch.int32).contiguous())
    gpu_mesh.fill_holes(max_hole_perimeter=3e-2)
    v, f = gpu_mesh.read()
    del gpu_mesh
    bvh = cumesh.cuBVH(v, f)
    v, f = cumesh.remeshing.remesh_narrow_band_dc(
        v, f, center=torch.zeros(3, device='cuda'),
        scale=(RESOLUTION + 3) / RESOLUTION, resolution=RESOLUTION,
        band=1, project_back=0, verbose=True, bvh=bvh)
    del bvh
    gc.collect()
    torch.cuda.empty_cache()
    gpu_mesh = cumesh.CuMesh()
    gpu_mesh.init(v.contiguous(), f.contiguous())
    del v, f
    torch.cuda.empty_cache()
    gpu_mesh.simplify(int(JOB['simplify_faces']), verbose=True)
    v, f = gpu_mesh.read()
    mesh = trimesh.Trimesh(v.cpu().numpy(), f.cpu().numpy(), process=False)
    # Y-up -> Z-up, puis plus grande dimension = SIZE mm.
    mesh.vertices = mesh.vertices[:, [0, 2, 1]] * np.array([1, -1, 1])
    mesh.apply_scale(SIZE / float(mesh.extents.max()))
    mesh.export(RAW_STL)
    mesh.export(OUT / f'{SLUG}.glb')
    record('mesh_ready', vertices=len(mesh.vertices), faces=len(mesh.faces), raw=raw_stats,
           watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
           bounds_mm=mesh.bounds.tolist(), elapsed_seconds=time.monotonic() - started,
           peak_cuda_allocated_gb=torch.cuda.max_memory_allocated() / 1024**3,
           stl=RAW_STL.name, print_ready=False)


def sample_distances(bvh, coords, resolution, scale, center, offset):
    import torch
    result = torch.empty(len(coords), dtype=torch.float32, device=coords.device)
    for start in range(0, len(coords), CHUNK):
        points = coords[start:start + CHUNK].float()
        if offset:
            points = points + offset
        points = (points / resolution - 0.5) * scale + center
        result[start:start + CHUNK] = bvh.unsigned_distance(points)[0]
    return result


def chunk_topology(coords, intersected, hashmap_vox, dual_verts, resolution, scale, center):
    import torch
    import cumesh.remeshing as module
    native = module.remesh_narrow_band_dc
    device = coords.device
    maximum = int(torch.count_nonzero(intersected).item())
    print(f'Topologie par blocs : {len(coords)} voxels, {maximum} aretes intersectees', flush=True)
    quads = torch.empty((maximum, 4), device=device, dtype=torch.int32)
    directions = torch.empty(maximum, device=device, dtype=torch.int32)
    cursor = 0
    for start in range(0, len(coords), CHUNK):
        part = intersected[start:start + CHUNK]
        mask = part != 0
        neighbors = coords[start:start + CHUNK, None, None, :] + native.edge_neighbor_voxel_offset
        connected = neighbors[mask]
        count = len(connected)
        if not count:
            continue
        keys = torch.cat([torch.zeros((count * 4, 1), dtype=torch.int32, device=device),
                          connected.reshape(-1, 3)], dim=1)
        indices = module._C.hashmap_lookup_3d_cuda(
            *hashmap_vox, keys, resolution, resolution, resolution).reshape(count, 4).int()
        valid = (indices != -1).all(dim=1)
        kept = indices[valid]
        count = len(kept)
        quads[cursor:cursor + count] = kept
        directions[cursor:cursor + count] = part[mask][valid]
        cursor += count
    quads = quads[:cursor]
    directions = directions[:cursor]
    used = torch.zeros(len(coords), dtype=torch.bool, device=device)
    for start in range(0, cursor, CHUNK):
        used[quads[start:start + CHUNK].reshape(-1)] = True
    unique = torch.nonzero(used, as_tuple=False).reshape(-1)
    vertices = (dual_verts[unique] / resolution - 0.5) * scale + center
    mapping = torch.empty(len(coords), dtype=torch.int32, device=device)
    mapping[unique] = torch.arange(len(unique), dtype=torch.int32, device=device)
    triangles = torch.empty((cursor * 2, 3), dtype=torch.int32, device=device)
    for start in range(0, cursor, CHUNK):
        q = mapping[quads[start:start + CHUNK]]
        direction = (directions[start:start + CHUNK] == 1).unsqueeze(1)
        best = []
        for split_p, split_n in ((native.quad_split_1_p, native.quad_split_1_n),
                                 (native.quad_split_2_p, native.quad_split_2_n)):
            attempt = torch.where(direction, q[:, split_p], q[:, split_n])
            n0 = torch.cross(vertices[attempt[:, 1]] - vertices[attempt[:, 0]],
                             vertices[attempt[:, 2]] - vertices[attempt[:, 0]], dim=1)
            n1 = torch.cross(vertices[attempt[:, 2]] - vertices[attempt[:, 1]],
                             vertices[attempt[:, 3]] - vertices[attempt[:, 1]], dim=1)
            best.append((attempt, (n0 * n1).sum(dim=1).abs()))
        (attempt0, align0), (attempt1, align1) = best
        selected = torch.where((align0 > align1).unsqueeze(1), attempt0, attempt1).reshape(-1, 3)
        triangles[start * 2:start * 2 + len(selected)] = selected
    return vertices, triangles


def patched_source(source):
    """Echoue proprement si la wheel installee n'a plus l'implementation attendue."""
    old = ('        pts = ((coords.float() + 0.5) / base_resolution - 0.5) * scale + center\n'
           '        \n        distances = bvh.unsigned_distance(pts)[0]')
    assert source.count(old) == 1
    source = source.replace(old, '        distances = _sample_distances(bvh, coords, base_resolution, scale, center, 0.5)')
    old = '        coords = coords[subdiv_mask]\n'
    assert source.count(old) == 1
    source = source.replace(old, old + '        del distances, subdiv_mask\n')
    old = ('    pts_vert = (grid_verts.float() / resolution - 0.5) * scale + center\n'
           '    distances_vert = bvh.unsigned_distance(pts_vert)[0]')
    assert source.count(old) == 1
    source = source.replace(old, '    distances_vert = _sample_distances(bvh, grid_verts, resolution, scale, center, 0)')
    start = source.index('    # Find connected voxels\n')
    end = source.index('    # --------------------------------------------------------------------------\n', start)
    source = source[:start] + (
        '    del hashmap_vert, grid_verts, distances_vert\n'
        '    mesh_vertices, mesh_triangles = _chunk_topology(\n'
        '        coords, intersected, hashmap_vox, dual_verts, resolution, scale, center)\n\n') + source[end:]
    compile(source, '<chunked-remeshing>', 'exec')
    return source


# --------------------------------------------------------------------------- etape reparation

def mesh_stats(mesh):
    import numpy as np
    counts = np.bincount(mesh.edges_unique_inverse)
    return dict(vertices=len(mesh.vertices), faces=len(mesh.faces),
                boundary_edges=int((counts == 1).sum()), nonmanifold_edges=int((counts > 2).sum()),
                watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
                valid_volume=bool(mesh.is_volume), volume_mm3=float(mesh.volume),
                extents_mm=mesh.extents.tolist())


def repair():
    import numpy as np
    import torch
    import trimesh
    import cumesh

    original = trimesh.load_mesh(RAW_STL, process=True)
    before = mesh_stats(original)
    record('cleaning', file='repair_report.json', before=before)
    # Les seuils CuMesh sont en coordonnees normalisees TRELLIS (taille ~1), pas en mm.
    gpu = cumesh.CuMesh()
    gpu.init(torch.tensor(np.asarray(original.vertices) / SIZE, dtype=torch.float32, device='cuda').contiguous(),
             torch.tensor(np.asarray(original.faces), dtype=torch.int32, device='cuda').contiguous())
    for _ in range(2):
        gpu.remove_duplicate_faces()
        gpu.repair_non_manifold_edges()
        gpu.remove_small_connected_components(1e-5)
        gpu.fill_holes(max_hole_perimeter=3e-2)
    gpu.unify_face_orientations()
    v, f = gpu.read()
    candidate = trimesh.Trimesh(v.cpu().numpy() * SIZE, f.cpu().numpy(), process=True)
    del gpu, v, f
    torch.cuda.empty_cache()
    trimesh.repair.fix_normals(candidate, multibody=True)
    candidate_path = OUT / f'{SLUG}-repair-candidate-{SIZE_TAG}.stl'
    candidate.export(candidate_path)
    # Valide le STL exact, apres fusion des sommets coincidents.
    delivered = trimesh.load_mesh(candidate_path, process=True)
    after = mesh_stats(delivered)
    np.random.seed(42)
    p0, _ = trimesh.sample.sample_surface(original, 3000)
    p1, _ = trimesh.sample.sample_surface(delivered, 3000)
    d0 = trimesh.proximity.closest_point(delivered, p0)[1]
    d1 = trimesh.proximity.closest_point(original, p1)[1]
    # Tolerances calibrees a 0.3 mm / 0.5 mm pour 100 mm, mises a l'echelle.
    p95_limit, extent_limit = 0.003 * SIZE, 0.005 * SIZE
    drift = dict(original_to_repair_p95_mm=float(np.percentile(d0, 95)),
                 repair_to_original_p95_mm=float(np.percentile(d1, 95)),
                 sampled_max_mm=float(max(d0.max(), d1.max())),
                 max_extent_change_mm=float(np.abs(original.extents - delivered.extents).max()),
                 p95_limit_mm=p95_limit, extent_limit_mm=extent_limit, samples_per_direction=3000)
    topology_ok = bool(delivered.is_volume and after['nonmanifold_edges'] == 0)
    shape_ok = bool(max(drift['original_to_repair_p95_mm'], drift['repair_to_original_p95_mm']) <= p95_limit
                    and drift['max_extent_change_mm'] <= extent_limit)
    if topology_ok and shape_ok:
        delivered.export(REPAIRED_STL)
        candidate_path.unlink(missing_ok=True)
    record('TOPOLOGY_AND_SAMPLED_DRIFT_PASS' if topology_ok and shape_ok else 'NEEDS_REVIEW',
           file='repair_report.json', before=before, after=after, drift=drift,
           topology_ok=topology_ok, sampled_shape_check_ok=shape_ok, print_validated=False,
           note="Pas de slicer, d'epaisseur, d'auto-intersection ni de test d'impression physique.")


# --------------------------------------------------------------------------- point d'entree

STAGES = dict(image=make_image, infer=infer, repair=repair)

if __name__ == '__main__':
    stage = os.environ.get('STL_STAGE')
    try:
        STAGES[stage]() if stage else bootstrap()
    except Exception as exc:
        traceback.print_exc()
        (OUT / f'error_{stage or "bootstrap"}.txt').write_text(traceback.format_exc())
        if stage != 'repair':
            record('failed', stage=stage or 'bootstrap', error=f'{type(exc).__name__}: {exc}')
        raise

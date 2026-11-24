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
    stop_after=None,             # 'image' : s'arrete apres reference.png (test SDXL)
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
    image_only = JOB['stop_after'] == 'image'
    if image_only and JOB['mode'] != 'text':
        raise ValueError("stop_after='image' n'a de sens qu'en mode texte")
    if not image_only:
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
    me = pathlib.Path(__file__).resolve()
    if image_only:
        # Test de l'etape SDXL seule : ni wheels, ni encodeur, ni TRELLIS.2.
        started = time.monotonic()
        run([python, '-u', me], env=dict(os.environ, STL_JOB=json.dumps(JOB), STL_STAGE='image'),
            timeout=1800)
        record('finished', file='summary.json', stopped_after='image',
               files=sorted(p.name for p in OUT.iterdir() if p.is_file()),
               total_seconds=round(time.monotonic() - started))
        return
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
    # TRELLIS.2 sort deja en Z vers le haut (verifie sur un rendu : pieds en bas). Plus grande
    # dimension = SIZE mm, objet centre en X/Y et pose sur le plateau (Z min = 0).
    mesh.apply_scale(SIZE / float(mesh.extents.max()))
    low, high = mesh.bounds
    mesh.apply_translation([-(low[0] + high[0]) / 2, -(low[1] + high[1]) / 2, -low[2]])
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


def pair_half_edges(vertices, faces):
    """Apparie chaque demi-arete a->b avec une demi-arete b->a de la meme arete.

    Demi-arete h = coin 3f+k : du sommet faces[f,k] vers faces[f,k+1].
    Aux aretes a 4 ou 6 faces (nappes qui se touchent), on apparie selon la geometrie : autour
    de l'axe a->b, la normale d'une face a->b pointe vers les angles croissants, donc sa matiere
    est fermee par la face b->a qui la precede dans l'ordre angulaire.
    Renvoie (fwd, bwd, multiple) : demi-aretes appariees, et vrai si l'arete avait plus de 2 faces.
    """
    import numpy as np
    he_from, he_to = faces.reshape(-1), faces[:, [1, 2, 0]].reshape(-1)
    third = faces[:, [2, 0, 1]].reshape(-1)
    lo, hi = np.minimum(he_from, he_to), np.maximum(he_from, he_to)
    forward = he_from < he_to
    order = np.lexsort((~forward, hi, lo))
    key = lo[order] * (len(vertices) + 1) + hi[order]
    start = np.r_[0, np.flatnonzero(np.diff(key)) + 1]
    count = np.diff(np.r_[start, len(key)])
    nfwd = np.add.reduceat(forward[order].astype(np.int64), start)
    out = [], [], []
    for c in np.unique(count):
        sel = start[(count == c) & (nfwd * 2 == c)]
        if c < 2 or not len(sel):
            continue
        half = c // 2
        he = order[sel[:, None] + np.arange(c)]              # (groupes, c) : a->b puis b->a
        fwd, bwd = he[:, :half].ravel(), he[:, half:].ravel()
        if c > 2:
            a, b = vertices[lo[he[:, 0]]], vertices[hi[he[:, 0]]]
            u = (b - a) / np.linalg.norm(b - a, axis=1, keepdims=True)
            e1 = np.cross(u, [0.577, 0.577, 0.577]); e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
            e2 = np.cross(u, e1)
            d = vertices[third[he]] - a[:, None]
            angle = np.arctan2((d * e2[:, None]).sum(-1), (d * e1[:, None]).sum(-1))
            ranked = np.argsort(angle, axis=1)
            previous = np.roll(ranked, 1, axis=1)
            rows = np.arange(len(he))[:, None]
            prev_of = np.take_along_axis(previous, ranked.argsort(1), 1)   # predecesseur de chaque face
            ok = np.repeat((prev_of[:, :half] >= half).all(1), half)
            is_fwd = ranked < half
            fwd = np.where(ok, he[rows, ranked][is_fwd], fwd)
            bwd = np.where(ok, he[rows, previous][is_fwd], bwd)
        out[0].append(fwd); out[1].append(bwd); out[2].append(np.full(len(fwd), c > 2))
    return [np.concatenate(x) for x in out]


def split_vertices(vertices, faces, gap):
    """Donne une copie de sommet par eventail de coins relies par des aretes appariees.

    Le STL n'ayant pas d'indices, chaque copie d'un sommet dedouble recule de `gap` vers
    l'interieur de sa nappe, sinon les slicers la refusionneraient.
    """
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(faces)
    fwd, bwd, _ = pair_half_edges(vertices, faces)
    nxt = lambda h: (h // 3) * 3 + (h % 3 + 1) % 3      # coin a l'extremite de la demi-arete
    # Demi-arete f: a->b, g: b->a. Coin de f en a == coin de g en a (extremite de g), idem en b.
    rows, cols = np.r_[fwd, nxt(fwd)], np.r_[nxt(bwd), bwd]
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(3 * n, 3 * n))
    ncomp, label = connected_components(graph, directed=False)
    original = faces.reshape(-1)
    new_vertices = np.empty((ncomp, 3)); new_vertices[label] = vertices[original]
    new_faces = label.reshape(n, 3)
    source = np.empty(ncomp, dtype=np.int64); source[label] = original
    split = np.bincount(source, minlength=len(vertices))[source] > 1
    tri = new_vertices[new_faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])   # ponderees par l'aire
    fan = np.zeros((ncomp, 3)); np.add.at(fan, new_faces.reshape(-1), np.repeat(normals, 3, axis=0))
    length = np.linalg.norm(fan, axis=1, keepdims=True)
    move = split & (length[:, 0] > 0)
    new_vertices[move] -= gap * fan[move] / length[move]
    return new_vertices, new_faces, int(split.sum())


def split_pinched_edges(vertices, faces, gap):
    """Aretes encore partagees par 4+ faces (sommets non dedoublables) : chaque paire de faces
    recoit son propre point milieu, en retrait vers l'interieur de sa nappe."""
    import numpy as np
    fwd, bwd, multiple = pair_half_edges(vertices, faces)
    fwd, bwd = fwd[multiple], bwd[multiple]
    if not len(fwd):
        return vertices, faces, 0
    def corners(h):
        f, k = h // 3, h % 3
        return f, faces[f, k], faces[f, (k + 1) % 3], faces[f, (k + 2) % 3]
    f, a, b, cf = corners(fwd)
    g, _, _, cg = corners(bwd)                           # g : b->a, sommet oppose cg
    # Un triangle touchant deux aretes pincees ne doit etre coupe qu'une fois par passe :
    # on ne garde que les paires dont les deux triangles sont libres (make_manifold repasse).
    used = np.bincount(np.r_[f, g], minlength=len(faces))
    free = (used[f] == 1) & (used[g] == 1)
    if not free.any():
        free[0] = True
    f, a, b, cf, g, cg = f[free], a[free], b[free], cf[free], g[free], cg[free]
    va, vb = vertices[a], vertices[b]
    nf = np.cross(vb - va, vertices[cf] - va); ng = np.cross(va - vb, vertices[cg] - vb)
    inward = (nf / np.linalg.norm(nf, axis=1, keepdims=True)
              + ng / np.linalg.norm(ng, axis=1, keepdims=True))
    inward /= np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-12)
    mid = np.arange(len(vertices), len(vertices) + len(f))
    vertices = np.vstack([vertices, (va + vb) / 2 - gap * inward])
    keep = np.ones(len(faces), bool); keep[f] = keep[g] = False
    faces = np.vstack([faces[keep],
                       np.c_[a, mid, cf], np.c_[mid, b, cf],      # f = (a, b, cf) coupee en deux
                       np.c_[b, mid, cg], np.c_[mid, a, cg]])     # g = (b, a, cg) coupee en deux
    return vertices, faces, len(f)


def make_manifold(vertices, faces, gap=0.002, min_fraction=0.001):
    """Retire les fragments, puis separe les nappes qui se touchent (aucun point ne bouge de
    plus de `gap`). Renvoie un maillage aux sommets fusionnes, comme relu depuis un STL."""
    import numpy as np
    import trimesh
    m = trimesh.Trimesh(vertices, faces, process=True)
    parts = m.split(only_watertight=False)
    biggest = max(len(p.faces) for p in parts)
    kept = [p for p in parts if len(p.faces) >= min_fraction * biggest]
    m = trimesh.util.concatenate(kept)
    v, f, moved = split_vertices(np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int64), gap)
    m = trimesh.Trimesh(v, f, process=True)                 # fusion comme a la relecture d'un STL
    pinched = 0
    for _ in range(10):                                     # jusqu'a ce qu'il ne reste rien
        v, f, count = split_pinched_edges(np.asarray(m.vertices), np.asarray(m.faces, dtype=np.int64), gap)
        m = trimesh.Trimesh(v, f, process=True)
        pinched += count
        if not count:
            break
    return m, dict(
        fragments_removed=len(parts) - len(kept), vertices_split=moved, edges_split=pinched)


def repair():
    import numpy as np
    import trimesh

    original = trimesh.load_mesh(RAW_STL, process=True)
    before = mesh_stats(original)
    record('cleaning', file='repair_report.json', before=before)
    # Le dual contouring laisse des aretes a 4 faces (nappes qui se touchent) ; les passes CuMesh
    # en doublaient le nombre. Reparation CPU : retrait des fragments, puis separation des nappes
    # de 2 microns. Teste sur un maillage TRELLIS 1024 : 19 786 aretes -> 0, etanche, 37 s.
    candidate, repair_info = make_manifold(np.asarray(original.vertices),
                                           np.asarray(original.faces, dtype=np.int64))
    if candidate.volume < 0:
        candidate.invert()
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
    after['repair'] = repair_info
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

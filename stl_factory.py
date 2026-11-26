#!/usr/bin/env python3
"""Lanceur STL Factory : envoie un travail sur Kaggle et recupere le STL.

Exemples :
  python stl_factory.py texte "an anatomical human skull" --name crane --size 100
  python stl_factory.py image photo.png --name vase --size 80
  python stl_factory.py statut
  python stl_factory.py recuperer --name crane

Prerequis : pip install kaggle, et le jeton API dans ~/.kaggle/kaggle.json.
"""
import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
KERNEL_SOURCE = HERE / 'kernel' / 'main.py'
MARKER = '# @@JOB@@'
# Kernels de preparation du mode hors ligne (sorties montees dans le kernel principal).
OFFLINE_KERNELS = {'prepare_env': 'stl-factory-env', 'prepare_models': 'stl-factory-models'}


def kaggle_user():
    if os.environ.get('KAGGLE_USERNAME'):
        return os.environ['KAGGLE_USERNAME']
    for path in (pathlib.Path.home() / '.kaggle/kaggle.json',
                 pathlib.Path.home() / '.config/kaggle/kaggle.json'):
        if path.is_file():
            return json.loads(path.read_text())['username']
    sys.exit("Utilisateur Kaggle introuvable : definissez KAGGLE_USERNAME ou ~/.kaggle/kaggle.json")


def kaggle(*args, dry=False, check=True):
    cmd = ['kaggle', *map(str, args)]
    print('$', ' '.join(cmd), flush=True)
    if dry:
        return ''
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if check and result.returncode != 0:
        sys.exit(f'Echec de la commande kaggle :\n{output}')
    return output if result.returncode == 0 else None


def offline_ready(user, args):
    """Vrai si --hors-ligne est demande et que les deux kernels de preparation ont termine.
    Par defaut : mode avec Internet (valide en production) ; le hors ligne calait au chargement
    des modeles le 2026-11-24 (diagnostic en cours)."""
    if not getattr(args, 'hors_ligne', False):
        return False
    return all(kernel_state(f'{user}/{slug}') == 'complete' for slug in OFFLINE_KERNELS.values())


def build_kernel(job, user, args, kernel=None, offline=False):
    """Copie main.py avec le travail integre, plus les metadonnees du kernel.
    Hors ligne : sans Internet, avec les sorties des kernels de preparation en entree."""
    source = KERNEL_SOURCE.read_text(encoding='utf-8')
    lines = [line for line in source.splitlines() if line.rstrip().endswith(MARKER)]
    if len(lines) != 1:
        sys.exit(f'Marqueur {MARKER} introuvable ou duplique dans {KERNEL_SOURCE}')
    embedded = f'JOB = json.loads({json.dumps(job, ensure_ascii=False)!r})  {MARKER}'
    source = source.replace(lines[0], embedded)

    kernel = kernel or args.kernel
    build = HERE / 'build' / kernel
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)
    (build / 'main.py').write_text(source, encoding='utf-8')
    datasets = [f'{user}/{args.wheels_dataset}', f'{user}/{args.encoder_dataset}']
    preparing = job['mode'] in OFFLINE_KERNELS
    items = [{**job, **item} for item in job.get('batch') or [{}]]
    if all(item.get('stop_after') == 'image' for item in items):
        datasets = []                      # images seules : aucun dataset requis
    if any(item['mode'] == 'image' for item in items):
        datasets.append(f'{user}/{args.input_dataset}')
    if preparing:
        datasets = []
    kernels = [f'{user}/{slug}' for slug in OFFLINE_KERNELS.values()] if offline else []
    metadata = dict(
        id=f'{user}/{kernel}', title=kernel.replace('-', ' ').title(),
        code_file='main.py', language='python', kernel_type='script', is_private=True,
        enable_gpu=not preparing, enable_tpu=False, enable_internet=not offline,
        dataset_sources=datasets, competition_sources=[], kernel_sources=kernels, model_sources=[])
    if not preparing:
        metadata['machine_shape'] = 'NvidiaTeslaT4'
    (build / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    return build


def upload_image(image, user, args, files=None):
    """Publie l'image (reference.png) ou les images d'un lot ({nom de fichier: chemin})
    comme nouvelle version d'un dataset prive."""
    folder = HERE / 'build' / args.input_dataset
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    for name, source in (files or {'reference.png': image}).items():
        target = folder / name
        if source.suffix.lower() == '.png':
            shutil.copy(source, target)
            continue
        try:
            from PIL import Image
        except ImportError:
            sys.exit('Image non PNG : installez Pillow (pip install pillow) ou convertissez-la en PNG')
        Image.open(source).convert('RGBA').save(target)
    dataset_id = f'{user}/{args.input_dataset}'
    (folder / 'dataset-metadata.json').write_text(json.dumps(dict(
        title=args.input_dataset.replace('-', ' ').title(), id=dataset_id,
        licenses=[dict(name='CC0-1.0')]), indent=2))
    if args.dry_run:
        kaggle('datasets', 'version', '-p', folder, '-m', 'nouvelle image', dry=True)
        return
    if kaggle('datasets', 'status', dataset_id, check=False) is None:
        kaggle('datasets', 'create', '-p', folder)          # prive par defaut
    else:
        kaggle('datasets', 'version', '-p', folder, '-m', f'{len(files or [image])} image(s)')
    for _ in range(60):                                      # attend le traitement Kaggle
        status = (kaggle('datasets', 'status', dataset_id, check=False) or '').lower()
        if 'ready' in status:
            return
        time.sleep(10)
    sys.exit("Le dataset d'entree n'est pas pret apres 10 minutes")


def kernel_state(kernel_id):
    text = (kaggle('kernels', 'status', kernel_id, check=False) or '').lower()
    for state in ('complete', 'error', 'cancel', 'running', 'queued'):
        if state in text:
            return state
    return text or 'inconnu'


def fetch(kernel_id, slug, out):
    folder = out / slug
    folder.mkdir(parents=True, exist_ok=True)
    kaggle('kernels', 'output', kernel_id, '-p', folder)
    batch = folder / 'batch_summary.json'
    if batch.is_file():                    # lot : un sous-dossier par figurine
        for item in json.loads(batch.read_text()):
            print(f"\n=== {item['name']} : {'ok' if item['ok'] else 'ECHEC'} ({item['seconds']} s)")
            report(folder / item['slug'], item['slug'])
        return
    report(folder, slug)


def report(folder, slug):
    stls = sorted(folder.rglob('*.stl'))
    for name in ('summary.json', 'repair_report.json', 'status.json'):
        path = next(folder.rglob(name), None)
        if path:
            data = json.loads(path.read_text())
            print(f"\n{name} : statut = {data.get('status')}")
            if data.get('error'):
                print('Erreur :', data['error'])
            break
    sheet = next(folder.rglob('planche.png'), None)
    if sheet:
        summary = json.loads(next(folder.rglob('summary.json')).read_text())
        job = summary.get('job', {})
        print(f'\nPlanche des variantes : {sheet}')
        print('Choisissez une graine, puis lancez la 3D sur cette image exacte :')
        for variant in sorted(sheet.parent.glob('variantes/graine-*.png')):
            print(f'  python stl_factory.py image "{variant}" --name {slug}'
                  f" --style {job.get('style', 'objet')} --size {job.get('size_mm', 100):g}")
        return
    print('\nFichiers STL :' if stls else '\nAucun STL produit ; voir les fichiers error_*.txt')
    for stl in stls:
        print(f'  {stl}  ({stl.stat().st_size / 1e6:.1f} Mo)')


def run_batch(args, user, kernel_id):
    """Lot : une seule installation Kaggle, puis chaque figurine dans son propre processus."""
    items = json.loads(args.fichier.read_text(encoding='utf-8'))
    batch, images = [], {}
    for item in items:
        slug = slug_of(argparse.Namespace(name=item['name'], source=None))
        job = dict(name=item['name'], seed=item.get('seed', 42))
        for key, field in (('size', 'size_mm'), ('socle', 'base_mm')):
            if key in item:
                job[field] = item[key]
        if 'image' in item:
            image = (args.fichier.parent / item['image']).expanduser()
            if not image.is_file():
                sys.exit(f'Image introuvable : {image}')
            job.update(mode='image', input_image=f'{slug}.png')
            images[f'{slug}.png'] = image
        else:
            job.update(mode='text', prompt=item['prompt'])
        batch.append(job)
    if images and args.variantes > 1:
        sys.exit('--variantes ne sert qu\'avec des prompts texte')
    common = dict(name=args.name, style=args.style, image_steps=args.steps, resolution=args.finesse,
                  relief_mm=args.relief,
                  size_mm=args.size or (32.0 if args.style == 'miniature' else 100.0))
    if args.socle is not None:
        common['base_mm'] = args.socle
    if args.variantes > 1:
        common.update(variants=args.variantes, stop_after='image')
    job = dict(common, mode='image' if images else 'text', batch=batch)
    if images:
        upload_image(None, user, args, files=images)
    build = build_kernel(job, user, args, offline=offline_ready(user, args))
    slug = slug_of(argparse.Namespace(name=args.name, source=None))
    print(f'Lot pret : {len(batch)} travaux, kernel {build}  (sortie : {slug})')
    kaggle('kernels', 'push', '-p', build, dry=args.dry_run)
    if args.dry_run or args.no_wait:
        print(f'Ensuite : python stl_factory.py recuperer --name "{args.name}"')
        return
    wait_and_fetch(kernel_id, slug, args.out)


def slug_of(args):
    # Meme regle que slugify() dans kernel/main.py.
    import re
    import unicodedata
    text = unicodedata.normalize('NFKD', args.name or getattr(args, 'source', '') or '')
    text = re.sub(r'[^a-z0-9]+', '-', text.encode('ascii', 'ignore').decode().lower()).strip('-')
    return text[:40].strip('-') or 'objet'


def main():
    parser = argparse.ArgumentParser(description='STL Factory : texte ou image -> STL via Kaggle')
    parser.add_argument('--kernel', default='stl-factory', help='slug du kernel Kaggle')
    parser.add_argument('--wheels-dataset', default='trellis2-compiled-wheels')
    parser.add_argument('--encoder-dataset', default='dinov3-official-encoder')
    parser.add_argument('--input-dataset', default='stl-factory-input')
    parser.add_argument('--out', type=pathlib.Path, default=HERE / 'resultats')
    parser.add_argument('--hors-ligne', action='store_true',
                        help="experimental : sans Internet, avec les sorties de preparer-hors-ligne")
    sub = parser.add_subparsers(dest='command', required=True)
    for command, help_text in (('texte', 'description en texte (anglais conseille)'),
                               ('image', "chemin d'une image de l'objet")):
        p = sub.add_parser(command)
        p.add_argument('source', help=help_text)
        p.add_argument('--name', help='nom des fichiers de sortie')
        p.add_argument('--size', type=float,
                       help='plus grande dimension en mm, socle non compris '
                            '(defaut : 32 en miniature, 100 sinon)')
        p.add_argument('--style', choices=['objet', 'miniature'], default='objet',
                       help='miniature : prompt figurine de jeu + socle rond')
        p.add_argument('--socle', type=float,
                       help='diametre du socle en mm, 0 = sans (defaut : 25 en miniature)')
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--steps', type=int, default=30, help="etapes SDXL (mode texte)")
        p.add_argument('--no-wait', action='store_true', help='ne pas attendre la fin')
        p.add_argument('--jusqua', choices=['image'], help="s'arreter apres une etape (test)")
        p.add_argument('--dry-run', action='store_true', help='preparer sans rien envoyer')
        p.add_argument('--finesse', type=int, choices=[1024, 1536], default=1024,
                       help='resolution de la forme (1536 : details plus fins, plus long)')
        p.add_argument('--relief', type=float, default=0.0,
                       help='relief fin guide par la texture, amplitude max en mm (ex. 0.05)')
        if command == 'texte':
            p.add_argument('--variantes', type=int, default=1,
                           help='N > 1 : genere N images (graines seed..seed+N-1) et une planche, '
                                'sans 3D, pour choisir la meilleure')
    p = sub.add_parser('lot', help='plusieurs figurines en une session Kaggle (fichier JSON)')
    p.add_argument('fichier', type=pathlib.Path,
                   help='liste JSON de {"name", "prompt"} ou {"name", "image"}, '
                        'avec en option "seed", "size", "socle" par figurine')
    p.add_argument('--name', required=True, help='nom du lot (dossier de resultats)')
    p.add_argument('--style', choices=['objet', 'miniature'], default='miniature')
    p.add_argument('--size', type=float, help='hauteur par defaut (32 en miniature)')
    p.add_argument('--socle', type=float, help='diametre du socle par defaut')
    p.add_argument('--variantes', type=int, default=1, help='N > 1 : planches seulement, sans 3D')
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--finesse', type=int, choices=[1024, 1536], default=1024)
    p.add_argument('--relief', type=float, default=0.0, help='relief fin guide par la texture (mm)')
    p.add_argument('--no-wait', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p = sub.add_parser('preparer-hors-ligne',
                       help="une fois : prepare venv et modeles (2 kernels CPU) pour tourner sans Internet")
    p.add_argument('--dry-run', action='store_true')
    sub.add_parser('statut')
    p = sub.add_parser('recuperer')
    p.add_argument('--name', required=True)
    args = parser.parse_args()

    user = kaggle_user()
    kernel_id = f'{user}/{args.kernel}'
    if args.command == 'preparer-hors-ligne':
        for mode, slug in OFFLINE_KERNELS.items():
            build = build_kernel(dict(mode=mode, name=slug), user, args, kernel=slug)
            kaggle('kernels', 'push', '-p', build, dry=args.dry_run)
        print('Suivi : python stl_factory.py statut (ou kaggle kernels status <user>/stl-factory-env, '
              '<user>/stl-factory-models). Ensuite, les lots tournent sans Internet.')
        return
    if args.command == 'statut':
        print(kernel_state(kernel_id))
        return
    if args.command == 'recuperer':
        fetch(kernel_id, slug_of(args), args.out)
        return

    if args.command == 'lot':
        return run_batch(args, user, kernel_id)
    size = args.size or (32.0 if args.style == 'miniature' else 100.0)
    job = dict(mode='text' if args.command == 'texte' else 'image', name=args.name,
               size_mm=size, seed=args.seed, image_steps=args.steps, style=args.style,
               resolution=args.finesse, relief_mm=args.relief)
    if args.socle is not None:
        job['base_mm'] = args.socle
    if getattr(args, 'variantes', 1) > 1:
        job['variants'] = args.variantes
        job['stop_after'] = 'image'        # la 3D se lance ensuite sur l'image choisie
    if args.jusqua:
        if args.command != 'texte':
            sys.exit('--jusqua image ne sert qu\'en mode texte')
        job['stop_after'] = args.jusqua
    if args.command == 'texte':
        job['prompt'] = args.source
    else:
        image = pathlib.Path(args.source).expanduser()
        if not image.is_file():
            sys.exit(f'Image introuvable : {image}')
        job['name'] = args.name or image.stem
        upload_image(image, user, args)
    slug = slug_of(argparse.Namespace(name=job['name'], source=job.get('prompt')))

    build = build_kernel(job, user, args, offline=offline_ready(user, args))
    print(f'Kernel pret : {build}  (sortie : {slug})')
    kaggle('kernels', 'push', '-p', build, dry=args.dry_run)
    if args.dry_run or args.no_wait:
        print(f'Ensuite : python stl_factory.py recuperer --name "{job["name"] or slug}"')
        return

    wait_and_fetch(kernel_id, slug, args.out)


def wait_and_fetch(kernel_id, slug, out):
    started = time.monotonic()
    time.sleep(30)
    while True:
        state = kernel_state(kernel_id)
        print(f'[{(time.monotonic() - started) / 60:5.1f} min] {state}', flush=True)
        if state in ('complete', 'error', 'cancel'):
            break
        time.sleep(60)
    fetch(kernel_id, slug, out)


if __name__ == '__main__':
    main()

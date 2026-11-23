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


def build_kernel(job, user, args):
    """Copie main.py avec le travail integre, plus les metadonnees du kernel."""
    source = KERNEL_SOURCE.read_text(encoding='utf-8')
    lines = [line for line in source.splitlines() if line.rstrip().endswith(MARKER)]
    if len(lines) != 1:
        sys.exit(f'Marqueur {MARKER} introuvable ou duplique dans {KERNEL_SOURCE}')
    embedded = f'JOB = json.loads({json.dumps(job, ensure_ascii=False)!r})  {MARKER}'
    source = source.replace(lines[0], embedded)

    build = HERE / 'build' / args.kernel
    shutil.rmtree(build, ignore_errors=True)
    build.mkdir(parents=True)
    (build / 'main.py').write_text(source, encoding='utf-8')
    datasets = [f'{user}/{args.wheels_dataset}', f'{user}/{args.encoder_dataset}']
    if job['mode'] == 'image':
        datasets.append(f'{user}/{args.input_dataset}')
    metadata = dict(
        id=f'{user}/{args.kernel}', title=args.kernel.replace('-', ' ').title(),
        code_file='main.py', language='python', kernel_type='script', is_private=True,
        enable_gpu=True, enable_tpu=False, enable_internet=True, machine_shape='NvidiaTeslaT4',
        dataset_sources=datasets, competition_sources=[], kernel_sources=[], model_sources=[])
    (build / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
    return build


def upload_image(image, user, args):
    """Publie l'image comme version d'un dataset prive contenant reference.png."""
    folder = HERE / 'build' / args.input_dataset
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    target = folder / 'reference.png'
    if image.suffix.lower() == '.png':
        shutil.copy(image, target)
    else:
        try:
            from PIL import Image
        except ImportError:
            sys.exit('Image non PNG : installez Pillow (pip install pillow) ou convertissez-la en PNG')
        Image.open(image).convert('RGBA').save(target)
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
        kaggle('datasets', 'version', '-p', folder, '-m', f'image {image.name}')
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
    stls = sorted(folder.rglob('*.stl'))
    for name in ('summary.json', 'repair_report.json', 'status.json'):
        report = next(folder.rglob(name), None)
        if report:
            data = json.loads(report.read_text())
            print(f"\n{name} : statut = {data.get('status')}")
            if data.get('error'):
                print('Erreur :', data['error'])
            break
    print('\nFichiers STL :' if stls else '\nAucun STL produit ; voir les fichiers error_*.txt')
    for stl in stls:
        print(f'  {stl}  ({stl.stat().st_size / 1e6:.1f} Mo)')


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
    sub = parser.add_subparsers(dest='command', required=True)
    for command, help_text in (('texte', 'description en texte (anglais conseille)'),
                               ('image', "chemin d'une image de l'objet")):
        p = sub.add_parser(command)
        p.add_argument('source', help=help_text)
        p.add_argument('--name', help='nom des fichiers de sortie')
        p.add_argument('--size', type=float, default=100.0, help='plus grande dimension en mm')
        p.add_argument('--seed', type=int, default=42)
        p.add_argument('--steps', type=int, default=30, help="etapes SDXL (mode texte)")
        p.add_argument('--no-wait', action='store_true', help='ne pas attendre la fin')
        p.add_argument('--dry-run', action='store_true', help='preparer sans rien envoyer')
    sub.add_parser('statut')
    p = sub.add_parser('recuperer')
    p.add_argument('--name', required=True)
    args = parser.parse_args()

    user = kaggle_user()
    kernel_id = f'{user}/{args.kernel}'
    if args.command == 'statut':
        print(kernel_state(kernel_id))
        return
    if args.command == 'recuperer':
        fetch(kernel_id, slug_of(args), args.out)
        return

    job = dict(mode='text' if args.command == 'texte' else 'image', name=args.name,
               size_mm=args.size, seed=args.seed, image_steps=args.steps)
    if args.command == 'texte':
        job['prompt'] = args.source
    else:
        image = pathlib.Path(args.source).expanduser()
        if not image.is_file():
            sys.exit(f'Image introuvable : {image}')
        job['name'] = args.name or image.stem
        upload_image(image, user, args)
    slug = slug_of(argparse.Namespace(name=job['name'], source=job.get('prompt')))

    build = build_kernel(job, user, args)
    print(f'Kernel pret : {build}  (sortie : {slug})')
    kaggle('kernels', 'push', '-p', build, dry=args.dry_run)
    if args.dry_run or args.no_wait:
        print(f'Ensuite : python stl_factory.py recuperer --name "{job["name"] or slug}"')
        return

    started = time.monotonic()
    time.sleep(30)
    while True:
        state = kernel_state(kernel_id)
        print(f'[{(time.monotonic() - started) / 60:5.1f} min] {state}', flush=True)
        if state in ('complete', 'error', 'cancel'):
            break
        time.sleep(60)
    fetch(kernel_id, slug, args.out)


if __name__ == '__main__':
    main()

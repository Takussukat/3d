#!/usr/bin/env python3
"""Publie un lot de figurines terminees comme dataset Kaggle prive (un dataset par jour).

Pour chaque figurine du lot (resultats/<lot>/<slug>/) : rendus produit (4 vues),
fiche.md, STL avec et sans socle, image de reference, planche des variantes. Les fichiers sont
a plat, prefixes par le slug, dans un dataset `<user>/figurines-AAAA-MM-JJ`.

  python outils/publier.py resultats/lot-2026-11-25 --date 2026-11-25 --planches resultats/planches-2026-11-25
  python outils/publier.py ... --dry-run       # prepare build/catalogue/<date> sans envoyer

Les descriptions de vente se placent dans <slug>.description.txt a cote du lot (optionnel) ;
sinon la fiche garde une section a completer.
"""
import argparse
import datetime
import json
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
RENDER = HERE / 'rendu' / 'rendu.mjs'


def figurines(lot):
    """(dossier, summary) de chaque figurine terminee du lot."""
    batch = next(lot.rglob('batch_summary.json'), None)
    base = batch.parent if batch else lot
    items = json.loads(batch.read_text()) if batch else [dict(slug=p.parent.name, ok=True)
                                                       for p in lot.rglob('summary.json')]
    for item in items:
        folder = base / item['slug']
        summary = folder / 'summary.json'
        if item.get('ok') and summary.is_file():
            yield folder, json.loads(summary.read_text())


def render(stl, out_dir, size=1200):
    """Rendus produit a partir d'une version allegee (300 k triangles) du STL."""
    import numpy as np
    import trimesh
    import fast_simplification
    mesh = trimesh.load_mesh(stl, process=True)          # process=True : sommets fusionnes
    light = out_dir / '_rendu.stl'
    if len(mesh.faces) > 300_000:
        v, f = fast_simplification.simplify(mesh.vertices.astype(np.float32),
                                            mesh.faces.astype(np.int32), target_count=300_000)
        mesh = trimesh.Trimesh(v, f, process=False)
    mesh.export(light)
    if not (HERE / 'rendu' / 'node_modules' / 'three').is_dir():
        subprocess.run(['npm', 'install', '--silent'], cwd=HERE / 'rendu', check=True)
    subprocess.run(['node', str(RENDER), str(light), str(out_dir), str(size)], check=True,
                   stdout=subprocess.DEVNULL)
    light.unlink()
    return sorted(out_dir.glob('rendu-*.png'))


def fiche(slug, summary, stl, with_base, description):
    import trimesh
    job = summary.get('job', {})
    mesh = trimesh.load_mesh(stl, process=True)
    x, y, z = mesh.extents
    import re
    diameter = re.search(r'-socle([0-9.]+)-', with_base.name).group(1) if with_base else None
    style = ('miniature de jeu, échelle héroïque 32 mm' if job.get('style') == 'miniature'
             else 'objet')
    lines = [
        f"# {job.get('name') or slug}",
        '',
        '## Description',
        '',
        description or '_À compléter : description de vente._',
        '',
        '## Caractéristiques',
        '',
        f'- Hauteur totale : {z:.1f} mm (figurine {job.get("size_mm", "?")} mm'
        + (f' + socle rond de {diameter} mm)' if diameter else ', sans socle)'),
        f'- Emprise : {x:.1f} × {y:.1f} mm',
        f'- Maillage : {len(mesh.faces):,} triangles'.replace(',', ' ')
        + f', étanche : {"oui" if mesh.is_watertight else "NON"}',
        f'- Type : {style}',
        '- Impression conseillée : résine, couche 0,05 mm ; supports à placer dans le slicer.',
        '',
        '## Fichiers',
        '',
        f'- `{slug}__figurine-socle.stl` : à imprimer (avec socle)' if with_base else None,
        f'- `{slug}__figurine.stl` : sans socle',
        f'- `{slug}__rendu-*.png` : images produit ; `{slug}__reference.png` : image IA d\'origine',
        '',
        '## Génération',
        '',
        f'- Prompt : `{job.get("prompt") or "(image fournie)"}`',
        f'- Graine : {job.get("seed")}, style : {job.get("style")}',
        '- Modèles : SDXL (OpenRAIL++-M), TRELLIS.2 (MIT), DINOv3 (licence DINOv3 de Meta), BiRefNet (MIT).',
        '- Non testé à l\'impression physique.',
    ]
    return '\n'.join(line for line in lines if line is not None) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('lot', type=pathlib.Path, help='dossier resultats du lot 3D')
    parser.add_argument('--date', default=datetime.date.today().isoformat())
    parser.add_argument('--planches', type=pathlib.Path, help='dossier resultats du lot de variantes')
    parser.add_argument('--user', required=True, help='utilisateur Kaggle')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    out = ROOT / 'build' / 'catalogue' / args.date
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    published = []
    for folder, summary in figurines(args.lot):
        slug = folder.name
        report_path = folder / 'repair_report.json'
        report = json.loads(report_path.read_text()) if report_path.is_file() else None
        with_base = next(folder.glob('*-socle*.stl'), None)
        repaired = next(folder.glob('*-repaired-*.stl'), None)
        main_stl = with_base or repaired
        if not main_stl:
            print(f'{slug} : pas de STL valide, ignoree', file=sys.stderr)
            continue
        if with_base:
            shutil.copy(with_base, out / f'{slug}__figurine-socle.stl')
        if repaired:
            shutil.copy(repaired, out / f'{slug}__figurine.stl')
        for name in ('reference.png', 'repair_report.json'):
            if (folder / name).is_file():
                shutil.copy(folder / name, out / f'{slug}__{name}')
        if args.planches:
            sheet = next(args.planches.rglob(f'{slug}/planche.png'), None)
            if sheet:
                shutil.copy(sheet, out / f'{slug}__planche-variantes.png')
        renders = out / f'_{slug}'
        renders.mkdir()
        for image in render(main_stl, renders):
            shutil.move(image, out / f'{slug}__{image.name}')
        renders.rmdir()
        text = folder.parent / f'{slug}.description.txt'
        description = text.read_text(encoding='utf-8').strip() if text.is_file() else None
        (out / f'{slug}__fiche.md').write_text(fiche(slug, summary, main_stl, with_base, description),
                                               encoding='utf-8')
        published.append(dict(slug=slug, name=summary.get('job', {}).get('name'),
                              prompt=summary.get('job', {}).get('prompt'),
                              seed=summary.get('job', {}).get('seed'), with_base=bool(with_base)))
        print(f'{slug} : pret')
    if not published:
        sys.exit('Aucune figurine valide a publier')

    dataset = f'{args.user}/figurines-{args.date}'
    (out / 'dataset-metadata.json').write_text(json.dumps(dict(
        title=f'Figurines {args.date}', id=dataset, licenses=[dict(name='other')],
        subtitle=f'{len(published)} figurines fantasy JDR generees (STL, rendus, fiches)'),
        indent=2))
    (out / 'index.json').write_text(json.dumps(published, indent=2, ensure_ascii=False))
    print(f'Dossier : {out}')
    if args.dry_run:
        return
    result = subprocess.run(['kaggle', 'datasets', 'create', '-p', str(out)],
                            capture_output=True, text=True)
    print(result.stdout[-500:], result.stderr[-500:])
    if result.returncode != 0 or 'successfully' not in (result.stdout + result.stderr).lower() \
            and 'being created' not in (result.stdout + result.stderr).lower():
        sys.exit('Echec de la creation du dataset Kaggle')
    history = ROOT / 'catalogue' / 'historique.csv'
    history.parent.mkdir(exist_ok=True)
    new = not history.is_file()
    with history.open('a', encoding='utf-8') as fh:
        if new:
            fh.write('date,slug,nom,graine,socle,dataset,prompt\n')
        for item in published:
            prompt = (item['prompt'] or '').replace('"', "'")
            fh.write(f'{args.date},{item["slug"]},"{item["name"]}",{item["seed"]},'
                     f'{"oui" if item["with_base"] else "non"},{dataset},"{prompt}"\n')
    print(f'Publie : https://www.kaggle.com/datasets/{dataset}')


if __name__ == '__main__':
    main()

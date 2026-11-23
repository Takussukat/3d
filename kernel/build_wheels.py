"""Kernel Kaggle (GPU T4) : compile les wheels CUDA utilisees par kernel/main.py.

Produit dans /kaggle/working/wheels : cumesh-*.whl, o_voxel-*.whl, flex_gemm-*.whl,
nvdiffrast-*.whl, a publier comme dataset prive `trellis2-compiled-wheels`.
Meme environnement que main.py : Python 3.10 (venv uv), torch 2.6.0 cu124.
Les commits sont figes : patched_source() de main.py depend de la source exacte de cumesh.
"""
import json
import os
import pathlib
import subprocess
import sys
import time

TRELLIS_REVISION = '75fbf0183001ed9876c8dbb35de6b68552ee08bd'
SOURCES = {
    'cumesh': ('https://github.com/JeffreyXiang/CuMesh.git',
               '12289e1062f0603f2f0d0771b02e1395d247f26f', ''),
    'flex_gemm': ('https://github.com/JeffreyXiang/FlexGEMM.git',
                  '6dd94a859c26ee8246888502eada3dd8ad85532e', ''),
    'o_voxel': ('https://github.com/microsoft/TRELLIS.2.git', TRELLIS_REVISION, 'o-voxel'),
    'nvdiffrast': ('https://github.com/NVlabs/nvdiffrast.git', 'v0.4.0', ''),
}

ROOT = pathlib.Path('/kaggle/temp/build-wheels')
OUT = pathlib.Path('/kaggle/working/wheels')
REPORT = pathlib.Path('/kaggle/working/build_report.json')
report = dict(status='starting', wheels={}, errors={})


def save(status):
    report['status'] = status
    REPORT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def run(args, **kwargs):
    print('+', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def output(args):
    return subprocess.run(list(map(str, args)), capture_output=True, text=True).stdout.strip()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    report['nvcc'] = output(['bash', '-lc', 'nvcc --version | tail -2'])
    report['gpu'] = output(['nvidia-smi', '--query-gpu=name,compute_cap', '--format=csv,noheader'])
    save('installing')

    run([sys.executable, '-m', 'pip', 'install', '-q', 'uv'])
    run([sys.executable, '-m', 'uv', 'venv', '--seed', '--python', '3.10', ROOT / 'venv'])
    python = ROOT / 'venv/bin/python'
    run([python, '-m', 'pip', 'install', '-q', 'torch==2.6.0', 'torchvision==0.21.0',
         '--index-url', 'https://download.pytorch.org/whl/cu124'])
    run([python, '-m', 'pip', 'install', '-q', 'setuptools>=64', 'wheel', 'ninja', 'numpy',
         'plyfile', 'trimesh', 'tqdm', 'zstandard', 'easydict'])
    report['torch'] = output([python, '-c', 'import torch; print(torch.__version__, torch.version.cuda)'])

    env = dict(os.environ, TORCH_CUDA_ARCH_LIST='7.5', MAX_JOBS=str(os.cpu_count() or 4),
               BUILD_TARGET='cuda')
    # Le venv doit fournir ninja et torch au build (--no-build-isolation).
    env['PATH'] = f"{ROOT / 'venv/bin'}:{env['PATH']}"
    save('building')
    for name, (url, revision, subdir) in SOURCES.items():
        started = time.monotonic()
        try:
            checkout = ROOT / 'src' / name
            if not checkout.exists():
                run(['git', 'clone', '--recursive', url, checkout])
            run(['git', 'checkout', revision], cwd=checkout)
            run(['git', 'submodule', 'update', '--init', '--recursive'], cwd=checkout)
            run([python, '-m', 'pip', 'wheel', '--no-build-isolation', '--no-deps', '-w', OUT,
                 checkout / subdir], env=env)
            built = sorted(p.name for p in OUT.glob(name + '-*.whl'))
            if len(built) != 1:
                raise RuntimeError(f'Attendu une wheel {name}-*.whl, trouve {built}')
            report['wheels'][name] = dict(file=built[0], revision=revision,
                                          seconds=round(time.monotonic() - started))
        except Exception as exc:
            report['errors'][name] = f'{type(exc).__name__}: {exc}'
        save('building')

    # Test d'import dans le venv, sur le GPU, avant publication.
    if not report['errors']:
        run([python, '-m', 'pip', 'install', '--no-deps', *sorted(OUT.glob('*.whl'))])
        check = ('import torch, cumesh, o_voxel, flex_gemm, nvdiffrast.torch as dr\n'
                 'import inspect, cumesh.remeshing as m\n'
                 'print("imports ok", torch.cuda.get_device_name(0), len(inspect.getsource(m)))\n'
                 'dr.RasterizeCudaContext()\nprint("nvdiffrast ok")')
        try:
            report['import_check'] = output([python, '-c', check]) or 'echec (voir le journal)'
            run([python, '-c', check])
        except Exception as exc:
            report['errors']['import_check'] = f'{type(exc).__name__}: {exc}'
    save('failed' if report['errors'] else 'success')
    if report['errors']:
        sys.exit(1)


if __name__ == '__main__':
    main()

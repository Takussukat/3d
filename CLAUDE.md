# STL Factory — contexte pour Claude Code

Pipeline texte/image → STL imprimable, exécuté sur Kaggle (GPU T4 16 Go, Python 3.10 en venv uv).
Répondre en français.

## Architecture
- `kernel/main.py` : kernel Kaggle en un seul fichier (Kaggle n'accepte qu'un fichier de code).
  Il se relance lui-même avec `STL_STAGE` = `image` → `infer` → `repair` pour libérer la VRAM entre les étapes.
  - `image` : SDXL + VAE fp16-fix → `reference.png` (mode texte uniquement)
  - `infer` : TRELLIS.2-4B, révision `75fbf018…`, cascade 512→1024, encodeur DINOv3 officiel
    lu depuis un dataset Kaggle privé (taille vérifiée), remeshing CuMesh patché par blocs (`patched_source`)
  - `repair` : CuMesh, puis contrôle de dérive (p95 ≤ 0,3 % de la taille) → `*-repaired-*.stl`
- `stl_factory.py` : lanceur local (CLI `kaggle`). Il intègre le travail dans la ligne `JOB = {}  # @@JOB@@`,
  pousse le kernel, attend la fin et télécharge les sorties dans `resultats/`.

## Contraintes
- Pas de GPU en local : le code de `kernel/main.py` ne s'exécute que sur Kaggle. En local, vérifier avec
  `python -m py_compile` et `python stl_factory.py texte "..." --dry-run`.
- Datasets privés requis : `<user>/trellis2-compiled-wheels` (wheels cumesh, o_voxel, flex_gemm, nvdiffrast)
  et `<user>/dinov3-official-encoder`.
- Ne pas toucher aux chaînes recherchées par `patched_source()` : elles doivent correspondre exactement
  à la source de la wheel cumesh installée.
- MeshFix plantait (segfault) à 1024 sur ce type de maillage : ne pas le réactiver sans test.
- Le pipeline n'a pas encore été validé de bout en bout sur Kaggle ; l'étape SDXL et le lanceur sont nouveaux.

## Idées suivantes
- Interface Gradio (lancement + aperçu 3D)
- File d'attente pour plusieurs travaux
- Vérification de l'épaisseur des parois avant impression

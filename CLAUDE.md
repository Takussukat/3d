# STL Factory — contexte pour Claude Code

Pipeline texte/image → STL imprimable, exécuté sur Kaggle (GPU T4 16 Go, Python 3.10 en venv uv).
Répondre en français.

## Architecture
- `kernel/main.py` : kernel Kaggle en un seul fichier (Kaggle n'accepte qu'un fichier de code).
  Il se relance lui-même avec `STL_STAGE` = `image` → `infer` → `repair` pour libérer la VRAM entre les étapes.
  - `image` : SDXL + VAE fp16-fix → `reference.png` (mode texte uniquement)
  - `infer` : TRELLIS.2-4B, révision `75fbf018…`, cascade 512→1024, encodeur DINOv3 officiel
    lu depuis un dataset Kaggle privé (taille vérifiée), remeshing CuMesh patché par blocs (`patched_source`)
  - `repair` (CPU) : retrait des fragments, puis `make_manifold` sépare de 2 µm les nappes qui se touchent
    (arêtes à 4 faces du dual contouring), puis contrôle de dérive (p95 ≤ 0,3 % de la taille) → `*-repaired-*.stl`.
    Les passes CuMesh doublaient les arêtes non-manifold : ne pas les remettre sans test.
  - Sortie TRELLIS.2 déjà en Z vers le haut : pas de rotation, objet posé sur Z = 0.
- `kernel/build_wheels.py` : kernel Kaggle qui compile les 4 wheels (CUDA 12.4 installé par le kernel, car l'image
  Kaggle a nvcc 13) pour le dataset `trellis2-compiled-wheels`.
- `stl_factory.py` : lanceur local (CLI `kaggle`). `--jusqua image` teste SDXL seul, sans dataset.
  `--style miniature` (prompt figurine + socle `add_base`, union manifold3d), `--variantes N` (planche d'images, sans 3D).
  Les suffixes de `STYLES` doivent rester courts : SDXL tronque à 75 tokens (miniature = 31 tokens). Il intègre le travail dans la ligne `JOB = {}  # @@JOB@@`,
  pousse le kernel, attend la fin et télécharge les sorties dans `resultats/`.

## Contraintes
- Pas de GPU en local : le code de `kernel/main.py` ne s'exécute que sur Kaggle. En local, vérifier avec
  `python -m py_compile` et `python stl_factory.py texte "..." --dry-run`.
- Datasets privés requis : `<user>/trellis2-compiled-wheels` (wheels cumesh, o_voxel, flex_gemm, nvdiffrast)
  et `<user>/dinov3-official-encoder`.
- Ne pas toucher aux chaînes recherchées par `patched_source()` : elles doivent correspondre exactement
  à la source de la wheel cumesh installée.
- MeshFix plantait (segfault) à 1024 sur ce type de maillage : ne pas le réactiver sans test.
- Validé de bout en bout sur Kaggle (troll, 23/11/2026) : ~12 min après installation, pic GPU 5,2 Go.
  La réparation CPU et l'orientation ont été validées en local sur ce maillage, pas encore dans un run Kaggle.
- Encodeur DINOv3 : copie publique `camenduru/dinov3-vitl16-pretrain-lvd1689m` (12 miroirs identiques ; empreinte
  différente du dépôt officiel actuel, accès officiel non obtenu). Licence DINOv3 de Meta.

## Idées suivantes
- Interface Gradio (lancement + aperçu 3D)
- File d'attente pour plusieurs travaux
- Vérification de l'épaisseur des parois avant impression

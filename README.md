# STL Factory

Texte ou image → modèle 3D → STL réparé, entièrement sur Kaggle (GPU T4).

```
texte ──SDXL──► reference.png ──TRELLIS.2 (512→1024)──► maillage ──CuMesh──► STL réparé
image ─────────►┘
```

## Contenu

| Fichier | Rôle |
|---|---|
| `kernel/main.py` | Le kernel Kaggle : installation, génération d'image, TRELLIS.2, réparation. Chaque étape tourne dans un processus séparé pour libérer la mémoire GPU. |
| `stl_factory.py` | Le lanceur, à exécuter sur votre PC ou dans Colab : il intègre le travail dans le kernel, le pousse sur Kaggle, attend la fin et télécharge les résultats. |

## Installation (une fois)

1. `pip install kaggle pillow`
2. Jeton API Kaggle : Settings → API → *Create New Token*, puis placez `kaggle.json` dans `~/.kaggle/`.
3. Gardez vos deux datasets privés existants : `trellis2-compiled-wheels` et `dinov3-official-encoder`.
   Si leurs noms changent, utilisez `--wheels-dataset` et `--encoder-dataset`.

## Utilisation

```bash
# Depuis un texte (prompts en anglais = meilleurs résultats)
python stl_factory.py texte "an anatomical human skull, ivory bone" --name crane --size 100

# Depuis une image (PNG/JPG ; un fond simple aide beaucoup)
python stl_factory.py image photo.jpg --name vase --size 80

# Envoyer sans attendre, puis récupérer plus tard
python stl_factory.py texte "a small dragon figurine" --name dragon --no-wait
python stl_factory.py statut
python stl_factory.py recuperer --name dragon
```

Options : `--seed` (autre variante), `--steps` (étapes SDXL), `--dry-run` (prépare sans rien envoyer).

### Figurines de jeu (`--style miniature`)

```bash
# 1. Six images (graines 42 à 47) et une planche d'aperçu, sans 3D (~5 min de GPU)
python stl_factory.py texte "a goblin archer with a short bow and a quiver" --name gobelin --style miniature --variantes 6
# 2. La 3D sur l'image choisie (le lanceur affiche la commande exacte pour chaque graine)
python stl_factory.py image resultats/gobelin/gobelin/variantes/graine-44.png --name gobelin --style miniature --size 32
```

- Le prompt est complété par un style « figurine 32 mm, sculpture en résine grise, silhouette lisible,
  détails épais », et le négatif écarte ce qui casse à l'impression (mèches fines, fumée, chaînes).
  Décrivez seulement le personnage, en anglais, en 40 mots environ : SDXL ignore tout après 75 tokens.
- `--size` = hauteur de la figurine sans socle (32 mm par défaut ; ~55–60 mm pour une grande créature).
- Socle rond biseauté de 3,2 mm, fusionné sous les pieds : 25 mm par défaut, `--socle 32`, `--socle 0` pour aucun.
  Il produit `<nom>-socle25-32mm.stl` en plus du STL réparé sans socle.

## Résultats (`resultats/<nom>/`)

- `reference.png`, `conditioning.png` : l'image générée, puis l'image détourée vue par TRELLIS.
- `<nom>-100mm.stl` et `<nom>.glb` : le maillage brut, mis à l'échelle.
- `<nom>-repaired-100mm.stl` : présent seulement si la réparation passe les contrôles (étanche, manifold, dérive ≤ 0,3 % de la taille).
- `repair_report.json`, `summary.json`, `status.json`, et `error_*.txt` en cas d'échec.

## Limites

- Testé de bout en bout sur Kaggle (troll de 60 mm : ~12 min après l'installation). La réparation CPU actuelle a été validée en local sur ce maillage : étanche, 0 arête non-manifold.
- Un seul travail à la fois par kernel : chaque envoi crée une nouvelle version de `stl-factory`. Pour en lancer plusieurs en parallèle, utilisez `--kernel stl-factory-2`.
- Compter ~1 à 2 h par objet (installation comprise) ; le quota GPU Kaggle est d'environ 30 h par semaine.
- « Réparé » ne veut pas dire « imprimable » : vérifiez l'épaisseur des parois et les surplombs dans votre slicer.
- SDXL (licence CreativeML OpenRAIL++-M) ; pour un autre modèle compatible SDXL, modifiez `image_model` dans `DEFAULTS`.
- Détourage : BiRefNet public (`ZhengPeng7/BiRefNet`, licence MIT). Le modèle prévu par Microsoft, `briaai/RMBG-2.0`, est sous licence non commerciale et n'est volontairement pas utilisé. Avant de vendre des STL, vérifiez aussi la licence de DINOv3.

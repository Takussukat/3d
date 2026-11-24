# Production quotidienne : 4 figurines fantasy JDR

Procédure suivie chaque jour par la session automatique (tâche planifiée). Tout se passe dans le
dépôt, sur la branche de travail. Durée : ~1 h 30, dont ~1 h 10 de GPU Kaggle.
`DATE` = date du jour (AAAA-MM-JJ). Les commandes longues se lancent en arrière-plan et se
suivent avec `python stl_factory.py statut` (ou les journaux : `kaggle kernels logs -f <user>/stl-factory`).

## 0. Préparation de la session

```bash
pip install -q kaggle trimesh fast-simplification manifold3d scipy pillow rtree
export PATH="$PWD/outils:$PATH" KAGGLE_USERNAME=yassineblhr   # outils/kaggle : CLI adaptée au proxy
(cd outils/rendu && npm install --silent)
kaggle kernels list --mine | head -3                         # vérifie l'accès
```

Si Kaggle répond 401/403 ou « no rule allows host », s'arrêter et le signaler : jeton ou réseau à
corriger dans l'environnement « 3d ».

## 1. Choisir 4 figurines

- Lire `catalogue/historique.csv` (déjà publiées) et `catalogue/idees.txt`.
- Prendre 4 concepts **jamais publiés**, de **catégories différentes** (par exemple un héros, un
  monstre, une grande créature, une bête ou un PNJ). Si la liste s'épuise, inventer des concepts du
  même genre et les ajouter à `idees.txt`. Jamais de personnage protégé (Warhammer, D&D « product
  identity », Pokémon, films…).
- Pour chacun, un prompt anglais de 30 mots maximum : le personnage, sa pose, son équipement.
  Pas de mots de style (le style `miniature` les ajoute). Préférer armes tenues près du corps,
  formes épaisses, pas de fumée, chaînes ni mèches fines.
- Taille : 32 mm pour un humanoïde, 45–60 mm pour une grande créature (`"size"`), avec un socle
  adapté (`"socle"` : 25 humanoïde, 32–40 grand, 50 très grand).

Écrire `build/jour/planches.json` :

```json
[{"name": "Gobelin archer", "prompt": "a goblin archer with a short bow and a quiver, aiming"},
 {"name": "Ogre massue", "prompt": "an ogre with a tree-trunk club, heavy belly", "size": 55, "socle": 40}]
```

## 2. Planches de variantes (~15 min)

```bash
python stl_factory.py lot build/jour/planches.json --name planches-DATE --variantes 4
```

Résultat : `resultats/planches-DATE/planches-DATE/<slug>/planche.png` et `variantes/graine-*.png`.

## 3. Choisir la meilleure variante de chaque figurine

Regarder chaque `planche.png`. Critères, dans l'ordre :
1. une seule figurine, entière (pieds et tête visibles), de face ou de trois-quarts ;
2. silhouette lisible, membres et armes épais, peu d'éléments fins détachés ;
3. anatomie correcte (mains, nombre de membres), arme cohérente ;
4. fond uni sans décor ni socle généré.

Si aucune variante n'est correcte, écarter la figurine du jour (3 au lieu de 4, le noter).
Écrire `build/jour/3d.json` avec le chemin absolu de l'image choisie et les mêmes `size`/`socle` :

```json
[{"name": "Gobelin archer", "image": "/home/user/3d/resultats/planches-DATE/planches-DATE/gobelin-archer/variantes/graine-44.png"}]
```

## 4. 3D, réparation, socle (~1 h)

```bash
python stl_factory.py lot build/jour/3d.json --name lot-DATE
```

Vérifier pour chaque figurine `repair_report.json` : statut `TOPOLOGY_AND_SAMPLED_DRIFT_PASS` et
`base.ok = true`. Une figurine en échec n'est pas publiée (la signaler dans le compte rendu).

## 5. Descriptions de vente

Pour chaque figurine réussie, écrire `resultats/lot-DATE/<slug>.description.txt` : 3 à 5 phrases
en français, vivantes, pour une fiche produit (qui est le personnage, son allure, son usage en
partie), sans promesse invérifiable (« testée à l'impression » est faux).

## 6. Publication sur Kaggle

```bash
python outils/publier.py resultats/lot-DATE --date DATE --planches resultats/planches-DATE --user yassineblhr
```

Crée le dataset privé `yassineblhr/figurines-DATE` (STL avec et sans socle, 4 rendus, fiche,
image d'origine, planche) et ajoute les lignes à `catalogue/historique.csv`. Regarder au moins un
rendu par figurine avant de conclure.

## 7. Historique dans Git

```bash
git add catalogue/historique.csv catalogue/idees.txt
git commit -m "Figurines du DATE : <noms>"
git push -u origin claude/lucid-turing-9tgcqa
```

Seuls ces petits fichiers vont dans Git ; les STL et images restent sur Kaggle.

## 8. Compte rendu

Message final : les figurines publiées (nom, hauteur, lien du dataset), celles écartées et
pourquoi, le temps GPU consommé, et une planche des rendus de face.

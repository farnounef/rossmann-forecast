# Rossmann Store Sales

Projet de prévision des ventes quotidiennes de magasins Rossmann, à partir des données de la compétition Kaggle [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales). J'ai utilisé DuckDB pour stocker et joindre les tables, LightGBM pour le modèle et MLflow pour suivre mes essais.

## Résultat

Je teste les modèles sur les 6 dernières semaines de l'historique (du 20 juin au 31 juillet 2015). Le modèle n'a jamais vu ces semaines pendant l'entraînement.

| Modèle | RMSPE |
|---|---|
| Baseline : moyenne des 4 dernières ventes du même jour de la semaine | 0,266 |
| LightGBM, 1000 arbres | 0,144 |

Le RMSPE est une erreur en pourcentage : 0,144 veut dire environ 14 % d'écart avec les ventes réelles. Le modèle réduit donc l'erreur de presque moitié par rapport à la prévision simple.

## Ce que j'ai fait

Je charge les trois fichiers CSV dans une base DuckDB et je joins `train` et `store` en SQL. Ensuite je retire les jours où le magasin est fermé, ainsi que les jours où il est ouvert mais sans aucune vente, qui ressemblent à des erreurs de saisie. Les jours fermés sont prédits à 0 à la fin.

Après quelques contrôles (doublons, valeurs manquantes, cohérence des colonnes Promo2, corrélations), je crée les variables du modèle :

- l'année, le mois, le jour et la semaine, tirés de la date ;
- le nombre de mois depuis l'ouverture du concurrent le plus proche ;
- le nombre de mois depuis le début de la dernière vague de Promo2 ;
- `StateHoliday`, `StoreType` et `Assortment` convertis en nombres.

Je n'utilise pas la colonne `Customers`, parce qu'on ne connaît pas le nombre de clients à l'avance.

Pour la validation, je coupe dans le temps (au 20 juin 2015) plutôt qu'au hasard : un découpage aléatoire mélangerait des jours voisins entre entraînement et test, et le score serait trop optimiste. Le modèle est entraîné sur le logarithme des ventes, puis je reviens aux vraies ventes pour les prédictions.

Chaque essai (baseline, deux versions de LightGBM, modèle final, prédictions) est enregistré dans MLflow. Le modèle final est réentraîné sur tout l'historique pour prédire `test.csv`. Le fichier de sortie garde toutes les lignes de `test.csv`.

## Organisation du dépôt

```
data/raw/            
Notebooks/
    rossmann.ipynb    
src/projet_home/
pyproject.toml
uv.lock
```

Les données, la base DuckDB et la base MLflow ne sont pas dans le dépôt (voir `.gitignore`).

## Lancer le projet

J'utilise [uv](https://docs.astral.sh/uv/) avec Python 3.13.

```bash
git clone https://github.com/<ton-nom>/<ton-depot>.git
cd <ton-depot>
uv sync
```

Les données sont à télécharger sur la [page Kaggle de la compétition](https://www.kaggle.com/c/rossmann-store-sales/data) et à placer dans `data/raw/`. Il suffit ensuite d'exécuter `Notebooks/rossmann.ipynb`, qui produit un fichier `submission.csv`.

Pour voir les essais dans MLflow, depuis la racine du projet :

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

L'interface est alors sur http://localhost:5000, dans l'expérience `rossmann`.


# Rossmann Store Sales



Projet de prévision des ventes quotidiennes de magasins Rossmann, à partir des données de la compétition Kaggle [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales). J'ai utilisé DuckDB pour stocker et joindre les tables, LightGBM pour le modèle, MLflow pour suivre mes essais et Evidently pour surveiller la dérive des données.

## Résultat

Je teste les modèles sur les 6 dernières semaines de l'historique (du 20 juin au 31 juillet 2015). Le modèle n'a jamais vu ces semaines pendant l'entraînement.

Le RMSPE est une erreur en pourcentage : 0,144 veut dire environ 14 % d'écart avec les ventes réelles. Avec le modèle à 1000 arbres, j'obtiens 0,144, contre 0,266 pour la prévision simple (moyenne des 4 dernières ventes du même jour de la semaine). Le modèle réduit donc l'erreur de presque moitié.

## Ce que j'ai fait

Je charge les trois fichiers CSV dans une base DuckDB et je joins `train` et `store` en SQL. Ensuite je retire les jours où le magasin est fermé, ainsi que les jours où il est ouvert mais sans aucune vente, qui ressemblent à des erreurs de saisie. Les jours fermés sont prédits à 0 à la fin.

Après quelques contrôles (doublons, valeurs manquantes, cohérence des colonnes Promo2, corrélations), je crée les variables du modèle :

- l'année, le mois, le jour et la semaine, tirés de la date ;
- le nombre de mois depuis l'ouverture du concurrent le plus proche ;
- le nombre de mois depuis le début de la dernière vague de Promo2 ;
- `StateHoliday`, `StoreType` et `Assortment` convertis en nombres ;
- des profils par magasin (niveau moyen de ventes, par jour de la semaine et selon la promo), calculés uniquement sur le passé.

Je n'utilise pas la colonne `Customers`, parce qu'on ne connaît pas le nombre de clients à l'avance.

Pour la validation, je coupe dans le temps (au 20 juin 2015) plutôt qu'au hasard : un découpage aléatoire mélangerait des jours voisins entre entraînement et test, et le score serait trop optimiste. Le modèle est entraîné sur le logarithme des ventes, puis je reviens aux vraies ventes pour les prédictions.

Je compare plusieurs versions de LightGBM, et la meilleure sur la validation est réentraînée sur tout l'historique pour prédire `test.csv`. Chaque essai (baseline, versions de LightGBM, modèle final, prédictions) est enregistré dans MLflow. Le fichier de sortie garde toutes les lignes de `test.csv`.

## Surveillance de la dérive des données

Avec Evidently, je compare les données d'entraînement et celles de `test.csv`, colonne par colonne, pour voir si elles se ressemblent. Les colonnes tirées de la date (`Year`, `Month`, `Day`, `WeekOfYear`) dérivent presque forcément, puisque le test couvre une autre période. Je produis donc deux rapports HTML : un complet, et un sans ces colonnes.

## Organisation du dépôt

```
data/raw/              train.csv, test.csv, store.csv (non publiés)
Notebooks/
    rossmann.ipynb     exploration, modèles, MLflow, prédictions
src/projet_home/       le pipeline en scripts Python
tests/                 tests automatiques (pytest)
.github/workflows/     intégration continue (GitHub Actions)
pyproject.toml
uv.lock
```

Les données, la base DuckDB, la base MLflow, les modèles sauvegardés et les rapports ne sont pas dans le dépôt (voir `.gitignore`).

## Lancer le projet

J'utilise [uv](https://docs.astral.sh/uv/) avec Python 3.13.

```bash
git clone https://github.com/farnounef/rossmann-forecast.git
cd rossmann-forecast
uv sync
```

Les données sont à télécharger sur la [page Kaggle de la compétition](https://www.kaggle.com/c/rossmann-store-sales/data) et à placer dans `data/raw/`.

Le plus simple est d'exécuter `Notebooks/rossmann.ipynb`, qui produit un fichier `submission.csv`. Le même pipeline existe aussi en scripts, à lancer depuis la racine du projet dans cet ordre :

```bash
uv run python -m projet_home.train
uv run python -m projet_home.predict
uv run python -m projet_home.drift
```

`train` compare les modèles et sauvegarde le meilleur, `predict` crée `submission.csv`, et `drift` écrit les rapports Evidently dans `reports/`.

Pour voir les essais dans MLflow, depuis la racine du projet :

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

L'interface est alors sur http://localhost:5000, dans l'expérience `rossmann`.

## Tests et intégration continue

```bash
uv run pytest
```

Les tests vérifient la création des variables, la métrique, la coupure dans le temps, la baseline et les profils par magasin. Ils tournent aussi automatiquement sur GitHub à chaque `push`, avec GitHub Actions.
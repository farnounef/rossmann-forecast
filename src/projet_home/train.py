"""Entraînement : validation dans le temps, baseline, trois LightGBM, suivi MLflow.
 uv run python -m projet_home.train
"""
import json

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd

from projet_home.config import (
    CUTOFF, EXPERIMENT_NAME, MLFLOW_URI, MODELS_DIR, PARAMS_V1, PARAMS_V2,
)
from projet_home.data import load_train_store
from projet_home.features import (
    FEATURES, FEATURES_V2, add_store_stats, build_features, clean, compute_store_stats,
)
from projet_home.metrics import rmspe

# Les essais comparés. use_stats : on ajoute les profils par magasin (FEATURES_V2)
EXPERIMENTS = [
    {"name": "lgbm_v1", "params": PARAMS_V1, "features": FEATURES, "use_stats": False},
    {"name": "lgbm_v2_1000_arbres", "params": PARAMS_V2, "features": FEATURES, "use_stats": False},
    {"name": "lgbm_v3_profils_magasin", "params": PARAMS_V2, "features": FEATURES_V2, "use_stats": True},
]


def split_by_date(df: pd.DataFrame):
    """Passé = entraînement, 6 dernières semaines = validation."""
    train_set = df[df["Date"] < CUTOFF]
    val_set = df[df["Date"] >= CUTOFF]
    assert train_set["Date"].max() < val_set["Date"].min()
    return train_set, val_set


def baseline_score(train_set: pd.DataFrame, val_set: pd.DataFrame) -> float:
    """Moyenne des 4 dernières ventes du même jour de la semaine, par magasin."""
    last4 = (
        train_set.sort_values("Date")
        .groupby(["Store", "DayOfWeek"]).tail(4)
        .groupby(["Store", "DayOfWeek"])["Sales"].mean()
        .rename("pred")
        .reset_index()
    )
    val_pred = val_set.merge(last4, on=["Store", "DayOfWeek"], how="left")
    val_pred["pred"] = val_pred["pred"].fillna(train_set["Sales"].mean())
    return rmspe(val_pred["Sales"], val_pred["pred"])


def run_experiment(exp: dict, train_set: pd.DataFrame, val_set: pd.DataFrame) -> float:
    """Entraîne sur le passé, note sur la validation, enregistre dans MLflow."""
    train_x, val_x = train_set, val_set
    if exp["use_stats"]:
        # Profils calculés sur le passé seulement, puis collés aux deux jeux
        stats = compute_store_stats(train_set)
        train_x = add_store_stats(train_set, stats)
        val_x = add_store_stats(val_set, stats)

    model = lgb.LGBMRegressor(**exp["params"])
    model.fit(train_x[exp["features"]], np.log1p(train_x["Sales"]))
    pred = np.expm1(model.predict(val_x[exp["features"]]))
    score = rmspe(val_x["Sales"], pred)

    with mlflow.start_run(run_name=exp["name"]):
        mlflow.log_params(exp["params"])
        mlflow.log_param("nb_features", len(exp["features"]))
        mlflow.log_param("profils_magasin", exp["use_stats"])
        mlflow.log_metric("rmspe", score)
        mlflow.lightgbm.log_model(model, "model", registered_model_name="rossmann-lgbm")
    return score


def train_final(df_feat: pd.DataFrame, exp: dict, val_score: float):
    """Réentraîne le meilleur modèle sur toutes les données et le sauvegarde."""
    stats = None
    df_x = df_feat
    if exp["use_stats"]:
        stats = compute_store_stats(df_feat)
        df_x = add_store_stats(df_feat, stats)

    model = lgb.LGBMRegressor(**exp["params"])
    model.fit(df_x[exp["features"]], np.log1p(df_x["Sales"]))

    with mlflow.start_run(run_name="lgbm_final_toutes_donnees"):
        mlflow.log_params(exp["params"])
        mlflow.log_param("base_du_modele", exp["name"])
        mlflow.log_metric("rmspe_validation_attendu", val_score)
        mlflow.lightgbm.log_model(model, "model", registered_model_name="rossmann-lgbm-final")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "final_model.joblib")
    joblib.dump(stats, MODELS_DIR / "store_stats.joblib")   # None si pas de profils
    (MODELS_DIR / "metadata.json").write_text(json.dumps({
        "base_du_modele": exp["name"],
        "features": exp["features"],
        "use_stats": exp["use_stats"],
        "params": exp["params"],
        "rmspe_validation": val_score,
    }, indent=2))


def main():
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df_feat = build_features(clean(load_train_store()))
    train_set, val_set = split_by_date(df_feat)
    print(f"Entraînement : {len(train_set):,} lignes | validation : {len(val_set):,} lignes")

    score_base = baseline_score(train_set, val_set)
    with mlflow.start_run(run_name="baseline_moyenne_4_derniers"):
        mlflow.log_param("methode", "moyenne des 4 dernières ventes du même jour de la semaine")
        mlflow.log_metric("rmspe", score_base)
    print(f"Baseline : {score_base:.4f}")

    scores = {}
    for exp in EXPERIMENTS:
        scores[exp["name"]] = run_experiment(exp, train_set, val_set)
        print(f"{exp['name']} : {scores[exp['name']]:.4f}")

    best = min(EXPERIMENTS, key=lambda e: scores[e["name"]])
    print(f"Meilleur modèle : {best['name']} -> réentraînement sur toutes les données")
    train_final(df_feat, best, scores[best["name"]])


if __name__ == "__main__":
    main()
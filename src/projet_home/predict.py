"""Prédictions sur test.csv et création de submission.csv.

Lancement : uv run python -m projet_home.predict  (après projet_home.train)
"""
import json

import joblib
import mlflow
import numpy as np

from projet_home.config import EXPERIMENT_NAME, MLFLOW_URI, MODELS_DIR, SUBMISSION_PATH
from projet_home.data import load_test
from projet_home.features import add_store_stats, build_features


def main():
    model = joblib.load(MODELS_DIR / "final_model.joblib")
    stats = joblib.load(MODELS_DIR / "store_stats.joblib")
    meta = json.loads((MODELS_DIR / "metadata.json").read_text())

    test = build_features(load_test())
    if meta["use_stats"]:
        test = add_store_stats(test, stats)
    is_open = test["Open"] == 1

    # Magasins fermés = 0, magasins ouverts = prédiction du modèle
    test["Sales"] = 0.0
    test.loc[is_open, "Sales"] = np.expm1(model.predict(test.loc[is_open, meta["features"]]))

    submission = test[["Id", "Sales"]]
    assert submission["Sales"].notna().all()
    assert (submission["Sales"] >= 0).all()
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"{len(submission):,} lignes écrites dans {SUBMISSION_PATH.name}")

    # Trace dans MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="predictions_test"):
        mlflow.log_params(meta["params"])
        mlflow.log_param("base_du_modele", meta["base_du_modele"])
        mlflow.log_metric("rmspe_validation", meta["rmspe_validation"])
        mlflow.log_metric("nb_lignes", len(submission))
        mlflow.log_metric("nb_magasins_fermes", int((~is_open).sum()))
        mlflow.log_metric("ventes_moyennes_predites", float(test.loc[is_open, "Sales"].mean()))
        mlflow.log_artifact(str(SUBMISSION_PATH))


if __name__ == "__main__":
    main()
"""Chemins et paramètres du projet."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw"
DUCKDB_PATH = ROOT / "rossmann.duckdb"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
SUBMISSION_PATH = ROOT / "submission.csv"

MLFLOW_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
EXPERIMENT_NAME = "rossmann"

CUTOFF = "2015-06-20"
RANDOM_STATE = 42

PARAMS_V1 = {"n_estimators": 500, "learning_rate": 0.1, "num_leaves": 63, "random_state": RANDOM_STATE}
PARAMS_V2 = {"n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 127, "random_state": RANDOM_STATE}
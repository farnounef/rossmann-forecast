"""API de prévision des ventes.

Lancement : uv run uvicorn projet_home.api:app --reload
Documentation interactive : http://127.0.0.1:8000/docs
"""
import json
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from projet_home.config import DATA_DIR, MODELS_DIR
from projet_home.features import add_store_stats, build_features


class Row(BaseModel):
    """Une journée d'un magasin à prédire."""
    Store: int = Field(ge=1, description="Numéro du magasin")
    Date: date
    Promo: int = Field(ge=0, le=1, description="1 si promotion ce jour-là")
    StateHoliday: Literal["0", "a", "b", "c"] = "0"
    SchoolHoliday: int = Field(default=0, ge=0, le=1)
    Open: int = Field(default=1, ge=0, le=1, description="0 si le magasin est fermé")


class PredictRequest(BaseModel):
    rows: list[Row] = Field(min_length=1, max_length=10_000)


def load_artifacts() -> dict:
    """Charge le modèle, les profils par magasin, ses métadonnées et store.csv."""
    try:
        return {
            "model": joblib.load(MODELS_DIR / "final_model.joblib"),
            "stats": joblib.load(MODELS_DIR / "store_stats.joblib"),
            "meta": json.loads((MODELS_DIR / "metadata.json").read_text()),
            "store": pd.read_csv(DATA_DIR / "store.csv"),
        }
    except FileNotFoundError as err:
        raise RuntimeError(
            f"Fichier introuvable : {err.filename}. "
            "Lance d'abord : uv run python -m projet_home.train"
        ) from err


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = load_artifacts()   # chargé une seule fois, au démarrage
    yield


app = FastAPI(title="Rossmann : prévision des ventes", lifespan=lifespan)


def predict_rows(rows: list[Row], artifacts: dict) -> list[float]:
    """Mêmes transformations que pour l'entraînement, puis prédiction."""
    store = artifacts["store"]
    df = pd.DataFrame([r.model_dump() for r in rows])
    df["Date"] = pd.to_datetime(df["Date"])
    df["DayOfWeek"] = df["Date"].dt.dayofweek + 1      # lundi = 1, comme dans Rossmann

    unknown = sorted(set(df["Store"]) - set(store["Store"]))
    if unknown:
        raise HTTPException(status_code=422, detail=f"Magasins inconnus : {unknown}")

    df = df.merge(store, on="Store", how="left")
    df = build_features(df)
    if artifacts["meta"]["use_stats"]:
        df = add_store_stats(df, artifacts["stats"])

    pred = np.expm1(artifacts["model"].predict(df[artifacts["meta"]["features"]]))
    pred = np.where(df["Open"] == 1, pred, 0.0)         # magasin fermé = 0
    return [round(float(p), 2) for p in pred]


@app.get("/")
def root():
    return {
        "message": "Rossmann Sales Forecasting API",
        "docs": "/docs",
        "health": "/health",
    }

@app.get("/health")
def health(request: Request):
    meta = request.app.state.artifacts["meta"]
    return {
        "status": "ok",
        "modele": meta["base_du_modele"],
        "rmspe_validation": meta["rmspe_validation"],
    }


@app.post("/predict")
def predict(body: PredictRequest, request: Request):
    sales = predict_rows(body.rows, request.app.state.artifacts)
    return {
        "predictions": [
            {"Store": r.Store, "Date": r.Date.isoformat(), "Sales": s}
            for r, s in zip(body.rows, sales)
        ]
    }
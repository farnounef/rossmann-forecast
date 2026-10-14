"""Nettoyage et construction des variables du projet Rossmann."""

import numpy as np
import pandas as pd

STATE_HOLIDAY = {"0": 0, "a": 1, "b": 2, "c": 3}
STORE_TYPE = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT = {"a": 0, "b": 1, "c": 2}

INTERVAL_MONTHS = {
    "Jan,Apr,Jul,Oct": [1, 4, 7, 10],
    "Feb,May,Aug,Nov": [2, 5, 8, 11],
    "Mar,Sept,Dec": [3, 9, 12],
}

FEATURES = [
    "Store", "DayOfWeek", "Promo", "SchoolHoliday", "StateHoliday",
    "Year", "Month", "Day", "WeekOfYear",
    "StoreType", "Assortment", "CompetitionDistance", "CompetitionMonthsOpen",
    "Promo2", "Promo2MonthsSinceWave",
]
STAT_FEATURES = ["store_mean", "store_dow_mean", "store_promo_mean"]
FEATURES_V2 = FEATURES + STAT_FEATURES


def clean(df):
    """Garde les jours où le magasin est ouvert et a vendu quelque chose."""
    return df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()


def build_features(df):
    """Ajoute le calendrier, les encodages, la concurrence et la vague Promo2."""
    df = df.copy()

    # Calendrier
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)

    # Lettres en chiffres (seulement si pas déjà fait)
    mappings = {
        "StateHoliday": STATE_HOLIDAY,
        "StoreType": STORE_TYPE,
        "Assortment": ASSORTMENT,
    }
    for col, mapping in mappings.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].map(mapping)

    # Concurrence : mois écoulés depuis l'ouverture du concurrent
    df["CompetitionMonthsOpen"] = (
        12 * (df["Year"] - df["CompetitionOpenSinceYear"])
        + (df["Month"] - df["CompetitionOpenSinceMonth"])
    ).clip(lower=0)

    # Promo2 : mois écoulés depuis le dernier démarrage de vague
    df["Promo2MonthsSinceWave"] = np.nan
    for interval, months in INTERVAL_MONTHS.items():
        mask = df["PromoInterval"] == interval
        if mask.any():
            gaps = np.column_stack([(df.loc[mask, "Month"] - m) % 12 for m in months])
            df.loc[mask, "Promo2MonthsSinceWave"] = gaps.min(axis=1)

    weeks_since_start = (
        52 * (df["Year"] - df["Promo2SinceYear"])
        + (df["WeekOfYear"] - df["Promo2SinceWeek"])
    )
    df.loc[weeks_since_start < 0, "Promo2MonthsSinceWave"] = np.nan

    # Types compatibles avec LightGBM
    for col in ["CompetitionDistance", "CompetitionMonthsOpen"]:
        df[col] = df[col].astype("float64")

    return df


def compute_store_stats(train):
    """Profils par magasin, calculés uniquement sur les données passées fournies."""
    hist = train[["Store", "DayOfWeek", "Promo", "Sales"]].copy()
    hist["log_sales"] = np.log1p(hist["Sales"])
    return {
        "store_mean": hist.groupby("Store")["log_sales"].mean()
                          .rename("store_mean").reset_index(),
        "store_dow_mean": hist.groupby(["Store", "DayOfWeek"])["log_sales"].mean()
                              .rename("store_dow_mean").reset_index(),
        "store_promo_mean": hist.groupby(["Store", "Promo"])["log_sales"].mean()
                                .rename("store_promo_mean").reset_index(),
    }


def add_store_stats(df, stats):
    """Colle les profils par magasin à chaque ligne."""
    out = df.merge(stats["store_mean"], on="Store", how="left")
    out = out.merge(stats["store_dow_mean"], on=["Store", "DayOfWeek"], how="left")
    out = out.merge(stats["store_promo_mean"], on=["Store", "Promo"], how="left")
    return out
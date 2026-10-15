"""Tests du découpage dans le temps, de la baseline et des profils par magasin."""
import numpy as np
import pandas as pd
import pytest

from projet_home.features import add_store_stats, compute_store_stats
from projet_home.metrics import rmspe
from projet_home.train import baseline_score, split_by_date




def test_split_by_date_has_no_overlap():
    df = pd.DataFrame({"Date": pd.date_range("2015-06-15", "2015-06-25")})
    train_set, val_set = split_by_date(df)
    assert train_set["Date"].max() < val_set["Date"].min()
    assert val_set["Date"].min() == pd.Timestamp("2015-06-20")
    assert len(train_set) + len(val_set) == len(df)


def test_baseline_is_zero_on_constant_sales():
    dates = pd.date_range("2015-05-01", periods=5, freq="7D")
    train_set = pd.DataFrame({"Store": 1, "DayOfWeek": 5, "Date": dates, "Sales": 100})
    val_set = pd.DataFrame({
        "Store": [1, 2], "DayOfWeek": [5, 5],
        "Date": pd.to_datetime(["2015-07-03", "2015-07-03"]),
        "Sales": [100, 100],
    })

    assert baseline_score(train_set, val_set) == 0


def test_store_stats_use_the_past_only():
    train = pd.DataFrame({
        "Store": [1, 1, 2], "DayOfWeek": [1, 1, 1], "Promo": [0, 0, 1],
        "Sales": [99, 99, 999],
    })
    stats = compute_store_stats(train)
    assert stats["store_mean"].set_index("Store").loc[1, "store_mean"] == pytest.approx(np.log1p(99))
    assert stats["store_mean"].set_index("Store").loc[2, "store_mean"] == pytest.approx(np.log1p(999))


def test_add_store_stats_keeps_rows_and_leaves_unknown_store_empty():
    train = pd.DataFrame({"Store": [1], "DayOfWeek": [1], "Promo": [0], "Sales": [99]})
    stats = compute_store_stats(train)
    new = pd.DataFrame({"Store": [1, 7], "DayOfWeek": [1, 1], "Promo": [0, 0]})
    out = add_store_stats(new, stats)
    assert len(out) == 2
    assert out.loc[0, "store_mean"] == pytest.approx(np.log1p(99))
    assert np.isnan(out.loc[1, "store_mean"])   # magasin jamais vu : valeur vide
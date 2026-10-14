import pandas as pd
from projet_home.features import clean
from projet_home.features import compute_store_stats
from projet_home.features import add_store_stats

def test_clean_keeps_only_open_stores_with_sales():
    df = pd.DataFrame({
        "Store": [1, 1, 2, 2],
        "Open": [1, 0, 1, 1],
        "Sales": [100, 200, 0, 300],
    })

    result = clean(df)

    assert len(result) == 2
    assert result.iloc[0]["Store"] == 1
    assert result.iloc[0]["Sales"] == 100
    assert (result["Open"] == 1).all()
    assert (result["Sales"] > 0).all()    



def test_compute_store_stats():
    train = pd.DataFrame({
        "Store": [1, 1, 2],
        "DayOfWeek": [1, 1, 2],
        "Promo": [0, 1, 0],
        "Sales": [100, 300, 200],
    })

    stats = compute_store_stats(train)

    assert set(stats.keys()) == {
        "store_mean",
        "store_dow_mean",
        "store_promo_mean",
    }
    assert len(stats["store_mean"]) == 2


def test_add_store_stats():
    train = pd.DataFrame({
        "Store": [1, 1, 2],
        "DayOfWeek": [1, 1, 2],
        "Promo": [0, 1, 0],
        "Sales": [100, 300, 200],
    })

    stats = compute_store_stats(train)
    result = add_store_stats(train, stats)

    assert len(result) == len(train)
    assert all(
        col in result.columns
        for col in ["store_mean", "store_dow_mean", "store_promo_mean"]
    )

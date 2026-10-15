"""Chargement des données : DuckDB pour le train, pandas pour le test."""
import duckdb
import pandas as pd

from projet_home.config import DATA_DIR, DUCKDB_PATH


def load_train_store() -> pd.DataFrame:
    """Charge train.csv et store.csv dans DuckDB, puis les joint."""
    train_csv = (DATA_DIR / "train.csv").as_posix()
    store_csv = (DATA_DIR / "store.csv").as_posix()

    con = duckdb.connect(str(DUCKDB_PATH))
    # StateHoliday est lu en texte : il mélange 0 et des lettres (a, b, c)
    con.execute(f"""
        CREATE OR REPLACE TABLE train AS
        SELECT * FROM read_csv_auto('{train_csv}', types={{'StateHoliday': 'VARCHAR'}})
    """)
    con.execute(f"""
        CREATE OR REPLACE TABLE store AS
        SELECT * FROM read_csv_auto('{store_csv}')
    """)
    df = con.execute("""
        SELECT t.*, s.* EXCLUDE (Store)
        FROM train t
        LEFT JOIN store s ON t.store = s.store
    """).df()
    con.close()
    return df


def load_test() -> pd.DataFrame:
    test = pd.read_csv(DATA_DIR / "test.csv", parse_dates=["Date"], dtype={"StateHoliday": str})
    store = pd.read_csv(DATA_DIR / "store.csv")
    test = test.merge(store, on="Store", how="left")
    # Open vide = on ne sait pas : on considère le magasin ouvert
    test["Open"] = test["Open"].fillna(1)
    return test
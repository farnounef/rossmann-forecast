"""Rapports de dérive des données avec Evidently (entraînement vs test).

Lancement : uv run python -m projet_home.drift
"""
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from projet_home.config import REPORTS_DIR
from projet_home.data import load_test, load_train_store
from projet_home.features import FEATURES, build_features, clean

NUM_COLS = ["Year", "Month", "Day", "WeekOfYear", "CompetitionDistance",
            "CompetitionMonthsOpen", "Promo2MonthsSinceWave"]
CAT_COLS = ["Store", "DayOfWeek", "Promo", "SchoolHoliday",
            "StateHoliday", "StoreType", "Assortment", "Promo2"]
DATE_COLS = ["Year", "Month", "Day", "WeekOfYear"]


def drift_report(reference_df, current_df, num_cols, cat_cols, output):
    definition = DataDefinition(numerical_columns=num_cols, categorical_columns=cat_cols)
    reference = Dataset.from_pandas(reference_df[num_cols + cat_cols], data_definition=definition)
    current = Dataset.from_pandas(current_df[num_cols + cat_cols], data_definition=definition)
    result = Report([DataDriftPreset()]).run(current, reference)  # d'abord le courant
    result.save_html(str(output))
    return result


def main():
    reference_df = build_features(clean(load_train_store()))[FEATURES]
    test = build_features(load_test())
    current_df = test[test["Open"] == 1][FEATURES]

    REPORTS_DIR.mkdir(exist_ok=True)
    drift_report(reference_df, current_df, NUM_COLS, CAT_COLS, REPORTS_DIR / "data_drift.html")

    # Même rapport sans les colonnes de date (elles dérivent forcément)
    num2 = [c for c in NUM_COLS if c not in DATE_COLS]
    drift_report(reference_df, current_df, num2, CAT_COLS, REPORTS_DIR / "data_drift_sans_dates.html")
    print(f"Rapports enregistrés dans {REPORTS_DIR}")


if __name__ == "__main__":
    main()
from pathlib import Path
from typing import Any

import pandas as pd

FEATURE_COLUMNS = (
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
)
TARGET_COLUMN = "HeartDisease"
EXPECTED_COLUMNS = (*FEATURE_COLUMNS, TARGET_COLUMN)

NUMERIC_COLUMNS = ("Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak")
CATEGORICAL_COLUMNS = (
    "Sex",
    "ChestPainType",
    "FastingBS",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
)

ZERO_AS_MISSING_CANDIDATES = ("RestingBP", "Cholesterol")


class DatasetSchemaError(ValueError):
    pass


def load_dataset(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Nie znaleziono pliku datasetu: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = sorted(set(EXPECTED_COLUMNS) - set(frame.columns))
    unexpected = sorted(set(frame.columns) - set(EXPECTED_COLUMNS))
    if missing or unexpected:
        raise DatasetSchemaError(
            "Niezgodny schemat datasetu. "
            f"Brakujące kolumny: {missing or 'brak'}; "
            f"nieoczekiwane kolumny: {unexpected or 'brak'}."
        )

    return frame.loc[:, EXPECTED_COLUMNS].copy()


def audit_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    missing = {column: int(count) for column, count in frame.isna().sum().items()}
    target_counts = {
        str(label): int(count)
        for label, count in frame[TARGET_COLUMN].value_counts(dropna=False).items()
    }
    suspicious_zeros = {
        column: int(frame[column].eq(0).sum()) for column in ZERO_AS_MISSING_CANDIDATES
    }

    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "duplicate_rows": int(frame.duplicated().sum()),
        "missing_values": missing,
        "target_distribution": target_counts,
        "suspicious_zeros": suspicious_zeros,
        "numeric_summary": frame.loc[:, NUMERIC_COLUMNS].describe().round(2).to_dict(),
        "categorical_values": {
            column: sorted(frame[column].dropna().astype(str).unique().tolist())
            for column in CATEGORICAL_COLUMNS
        },
    }

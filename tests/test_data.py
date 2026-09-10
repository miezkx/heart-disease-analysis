import pandas as pd
import pytest

from cardio_ml.data import (
    EXPECTED_COLUMNS,
    DatasetSchemaError,
    audit_dataset,
    load_dataset,
)


def sample_row() -> dict[str, object]:
    return {
        "Age": 40,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 140,
        "Cholesterol": 289,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 172,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up",
        "HeartDisease": 0,
    }


def test_load_dataset_accepts_expected_schema(tmp_path):
    path = tmp_path / "heart.csv"
    pd.DataFrame([sample_row()]).to_csv(path, index=False)

    frame = load_dataset(path)

    assert tuple(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 1


def test_load_dataset_rejects_missing_column(tmp_path):
    path = tmp_path / "heart.csv"
    row = sample_row()
    row.pop("MaxHR")
    pd.DataFrame([row]).to_csv(path, index=False)

    with pytest.raises(DatasetSchemaError, match="MaxHR"):
        load_dataset(path)


def test_audit_reports_duplicates_and_suspicious_zeros():
    row = sample_row()
    row["RestingBP"] = 0
    frame = pd.DataFrame([row, row])

    report = audit_dataset(frame)

    assert report["rows"] == 2
    assert report["duplicate_rows"] == 1
    assert report["suspicious_zeros"]["RestingBP"] == 2
    assert report["target_distribution"] == {"0": 2}


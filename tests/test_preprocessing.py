import numpy as np
import pandas as pd

from cardio_ml.preprocessing import build_preprocessor, replace_invalid_measurement_zeros


def feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [40, "M", "ATA", 140, 289, 0, "Normal", 172, "N", 0.0, "Up"],
            [49, "F", "NAP", 0, 0, 1, "ST", 156, "N", 1.0, "Flat"],
        ],
        columns=[
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
        ],
    )


def test_invalid_measurement_zeros_become_missing():
    cleaned = replace_invalid_measurement_zeros(feature_frame())

    assert np.isnan(cleaned.loc[1, "RestingBP"])
    assert np.isnan(cleaned.loc[1, "Cholesterol"])
    assert cleaned.loc[0, "Oldpeak"] == 0


def test_preprocessor_imputes_and_encodes_all_values():
    transformed = build_preprocessor().fit_transform(feature_frame())
    values = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    assert transformed.shape[0] == 2
    assert np.isfinite(values).all()

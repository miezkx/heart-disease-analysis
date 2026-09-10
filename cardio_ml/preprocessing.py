import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from cardio_ml.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def replace_invalid_measurement_zeros(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in ("RestingBP", "Cholesterol"):
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].replace(0, np.nan)
    return cleaned


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            (
                "zero_as_missing",
                FunctionTransformer(
                    replace_invalid_measurement_zeros,
                    feature_names_out="one-to-one",
                ),
            ),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(NUMERIC_COLUMNS)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_COLUMNS)),
        ]
    )

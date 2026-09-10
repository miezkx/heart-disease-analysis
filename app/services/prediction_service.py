import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.prediction.forms import FIELD_LABELS
from cardio_ml.data import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS


def predict_patient(
    patient: dict[str, Any],
    reference_frame: pd.DataFrame,
    model_path: str | Path,
    metadata_path: str | Path,
) -> dict[str, Any]:
    model_file = Path(model_path)
    metadata_file = Path(metadata_path)
    model = _load_model(str(model_file.resolve()), model_file.stat().st_mtime_ns)
    metadata = _load_metadata(str(metadata_file.resolve()), metadata_file.stat().st_mtime_ns)
    threshold = float(metadata.get("decision_threshold", 0.5))

    patient_frame = pd.DataFrame([patient], columns=FEATURE_COLUMNS)
    probability = float(model.predict_proba(patient_frame)[0, 1])
    predicted_class = int(probability >= threshold)
    reference = _reference_profile(reference_frame)

    return {
        "predicted_class": predicted_class,
        "probability": probability,
        "probability_percent": probability * 100,
        "threshold": threshold,
        "model_name": _display_model_name(metadata.get("selected_model", "model")),
        "model_version": metadata.get("model_version", "brak wersji"),
        "explanations": _local_perturbation_explanation(
            model, patient, reference, probability
        ),
    }


@lru_cache(maxsize=4)
def _load_model(path: str, modified_at: int) -> Any:
    del modified_at
    return joblib.load(path)


@lru_cache(maxsize=4)
def _load_metadata(path: str, modified_at: int) -> dict[str, Any]:
    del modified_at
    content = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("Nieprawidłowa struktura metadanych modelu.")
    return content


def _reference_profile(frame: pd.DataFrame) -> dict[str, Any]:
    reference: dict[str, Any] = {}
    for column in NUMERIC_COLUMNS:
        values = frame[column]
        if column in {"RestingBP", "Cholesterol"}:
            values = values.replace(0, pd.NA)
        reference[column] = float(values.median())
    for column in CATEGORICAL_COLUMNS:
        reference[column] = frame[column].mode(dropna=True).iloc[0]
    return reference


def _local_perturbation_explanation(
    model: Any,
    patient: dict[str, Any],
    reference: dict[str, Any],
    original_probability: float,
) -> list[dict[str, Any]]:
    impacts = []
    for feature in FEATURE_COLUMNS:
        changed = patient.copy()
        changed[feature] = reference[feature]
        changed_frame = pd.DataFrame([changed], columns=FEATURE_COLUMNS)
        changed_probability = float(model.predict_proba(changed_frame)[0, 1])
        impact = original_probability - changed_probability
        impacts.append(
            {
                "feature": feature,
                "label": FIELD_LABELS[feature],
                "value": _format_value(patient[feature]),
                "reference": _format_value(reference[feature]),
                "impact": impact,
                "impact_points": abs(impact) * 100,
                "direction": "zwiększał" if impact >= 0 else "zmniejszał",
            }
        )
    return sorted(impacts, key=lambda item: abs(item["impact"]), reverse=True)[:4]


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _display_model_name(value: str) -> str:
    return {
        "logistic_regression": "Regresja logistyczna",
        "decision_tree": "Drzewo decyzyjne",
        "random_forest": "Random Forest",
    }.get(value, value)

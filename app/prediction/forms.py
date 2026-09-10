from collections.abc import Mapping
from typing import Any

from cardio_ml.data import FEATURE_COLUMNS

NUMERIC_FIELDS = {
    "Age": {"label": "Wiek", "type": int, "minimum": 28, "maximum": 77},
    "RestingBP": {
        "label": "Ciśnienie spoczynkowe",
        "type": int,
        "minimum": 80,
        "maximum": 200,
    },
    "Cholesterol": {
        "label": "Cholesterol",
        "type": int,
        "minimum": 85,
        "maximum": 603,
    },
    "MaxHR": {
        "label": "Maksymalne tętno",
        "type": int,
        "minimum": 60,
        "maximum": 202,
    },
    "Oldpeak": {
        "label": "Oldpeak",
        "type": float,
        "minimum": -2.6,
        "maximum": 6.2,
    },
}

CATEGORICAL_FIELDS = {
    "Sex": {"label": "Płeć", "choices": ("F", "M")},
    "ChestPainType": {"label": "Typ bólu", "choices": ("ASY", "ATA", "NAP", "TA")},
    "FastingBS": {"label": "Cukier na czczo > 120 mg/dl", "choices": (0, 1)},
    "RestingECG": {"label": "EKG spoczynkowe", "choices": ("LVH", "Normal", "ST")},
    "ExerciseAngina": {"label": "Dusznica wysiłkowa", "choices": ("N", "Y")},
    "ST_Slope": {"label": "Nachylenie odcinka ST", "choices": ("Down", "Flat", "Up")},
}

FIELD_LABELS = {
    **{key: spec["label"] for key, spec in NUMERIC_FIELDS.items()},
    **{key: spec["label"] for key, spec in CATEGORICAL_FIELDS.items()},
}


def parse_patient_input(
    form: Mapping[str, str],
) -> tuple[dict[str, Any] | None, dict[str, str]]:
    parsed: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for field, spec in NUMERIC_FIELDS.items():
        raw_value = str(form.get(field, "")).strip().replace(",", ".")
        if not raw_value:
            errors[field] = "To pole jest wymagane."
            continue
        try:
            value = spec["type"](raw_value)
        except ValueError:
            errors[field] = "Wpisz prawidłową wartość liczbową."
            continue
        if not spec["minimum"] <= value <= spec["maximum"]:
            errors[field] = f"Dozwolony zakres: {spec['minimum']}–{spec['maximum']}."
            continue
        parsed[field] = value

    for field, spec in CATEGORICAL_FIELDS.items():
        raw_value = str(form.get(field, "")).strip()
        choices = spec["choices"]
        value: str | int = (
            int(raw_value)
            if field == "FastingBS" and raw_value in {"0", "1"}
            else raw_value
        )
        if value not in choices:
            errors[field] = "Wybierz jedną z dostępnych wartości."
            continue
        parsed[field] = value

    if errors:
        return None, errors
    return {column: parsed[column] for column in FEATURE_COLUMNS}, {}

import numpy as np
import pandas as pd

from app.prediction.forms import parse_patient_input
from app.services.prediction_service import _local_perturbation_explanation


def valid_form() -> dict[str, str]:
    return {
        "Age": "54",
        "Sex": "M",
        "ChestPainType": "ASY",
        "RestingBP": "140",
        "Cholesterol": "250",
        "FastingBS": "0",
        "RestingECG": "Normal",
        "MaxHR": "130",
        "ExerciseAngina": "Y",
        "Oldpeak": "1,5",
        "ST_Slope": "Flat",
    }


class FakeProbabilityModel:
    def predict_proba(self, frame):
        probability = 0.2 + frame["Age"].to_numpy() / 200 + frame["FastingBS"].to_numpy() * 0.1
        return np.column_stack([1 - probability, probability])


def test_valid_form_returns_typed_model_input():
    patient, errors = parse_patient_input(valid_form())

    assert errors == {}
    assert patient["Age"] == 54
    assert patient["Oldpeak"] == 1.5
    assert patient["FastingBS"] == 0


def test_invalid_form_reports_range_and_choice_errors():
    form = valid_form()
    form["Age"] = "100"
    form["Sex"] = "X"

    patient, errors = parse_patient_input(form)

    assert patient is None
    assert set(errors) == {"Age", "Sex"}


def test_local_explanation_orders_features_by_absolute_impact():
    patient, _ = parse_patient_input(valid_form())
    reference = patient.copy()
    reference["Age"] = 40
    reference["FastingBS"] = 1
    model = FakeProbabilityModel()
    probability = float(model.predict_proba(pd.DataFrame([patient]))[0, 1])

    explanation = _local_perturbation_explanation(model, patient, reference, probability)

    assert len(explanation) == 4
    assert explanation[0]["feature"] == "FastingBS"
    assert abs(explanation[0]["impact"]) >= abs(explanation[1]["impact"])


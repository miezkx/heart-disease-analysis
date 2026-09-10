import pandas as pd

from app.services.analysis_service import AnalysisFilters, apply_filters, build_analysis_context


def analysis_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            [40, "M", "ATA", 140, 289, 0, "Normal", 172, "N", 0.0, "Up", 0],
            [60, "M", "ASY", 150, 250, 1, "ST", 120, "Y", 2.0, "Flat", 1],
            [55, "F", "NAP", 130, 0, 0, "Normal", 145, "N", 0.5, "Up", 1],
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
            "HeartDisease",
        ],
    )


def test_apply_filters_combines_conditions():
    filtered = apply_filters(
        analysis_frame(),
        AnalysisFilters(age_min=50, sex="M", heart_disease=1),
    )

    assert filtered["Age"].tolist() == [60]


def test_analysis_context_excludes_zero_from_cholesterol_mean():
    context = build_analysis_context(analysis_frame(), {"heart_disease": "1"})

    assert context["summary"]["rows"] == 2
    assert context["summary"]["mean_cholesterol"] == 250.0
    assert context["filters"].active_count == 1


def test_invalid_filter_is_reported_and_ignored():
    context = build_analysis_context(analysis_frame(), {"sex": "X"})

    assert context["summary"]["rows"] == 3
    assert context["errors"]


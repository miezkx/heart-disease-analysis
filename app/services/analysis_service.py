from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import ceil
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html

from cardio_ml.data import EXPECTED_COLUMNS, TARGET_COLUMN

COLOR_HEALTHY = "#2f6b84"
COLOR_DISEASE = "#db5f57"
PAGE_SIZE = 25


@dataclass(frozen=True)
class AnalysisFilters:
    age_min: int | None = None
    age_max: int | None = None
    sex: str | None = None
    chest_pain: str | None = None
    exercise_angina: str | None = None
    heart_disease: int | None = None

    @property
    def active_count(self) -> int:
        return sum(value is not None for value in asdict(self).values())

    def query_parameters(self) -> dict[str, str | int]:
        return {key: value for key, value in asdict(self).items() if value is not None}


def build_analysis_context(
    frame: pd.DataFrame,
    query: Mapping[str, str],
    requested_page: int = 1,
) -> dict[str, Any]:
    filters, errors = parse_filters(query, frame)
    filtered = apply_filters(frame, filters)
    total_rows = len(filtered)
    total_pages = max(1, ceil(total_rows / PAGE_SIZE))
    page = min(max(requested_page, 1), total_pages)
    start = (page - 1) * PAGE_SIZE
    page_frame = filtered.iloc[start : start + PAGE_SIZE]

    cholesterol = filtered["Cholesterol"].replace(0, pd.NA).dropna()
    positive_rate = float(filtered[TARGET_COLUMN].mean()) if total_rows else None

    return {
        "filters": filters,
        "errors": errors,
        "options": {
            "age_min": int(frame["Age"].min()),
            "age_max": int(frame["Age"].max()),
            "sex": sorted(frame["Sex"].unique().tolist()),
            "chest_pain": sorted(frame["ChestPainType"].unique().tolist()),
            "exercise_angina": sorted(frame["ExerciseAngina"].unique().tolist()),
        },
        "summary": {
            "rows": total_rows,
            "all_rows": len(frame),
            "positive_rate": positive_rate,
            "mean_age": _optional_mean(filtered["Age"]),
            "mean_cholesterol": _optional_mean(cholesterol),
            "mean_max_hr": _optional_mean(filtered["MaxHR"]),
        },
        "charts": {
            "scatter": _chart_html(_age_max_hr_scatter(filtered)),
            "target": _chart_html(_target_distribution(filtered)),
            "cholesterol": _chart_html(_cholesterol_distribution(filtered)),
        },
        "table": {
            "columns": list(EXPECTED_COLUMNS),
            "rows": page_frame.to_dict(orient="records"),
        },
        "pagination": {
            "page": page,
            "total_pages": total_pages,
            "has_previous": page > 1,
            "has_next": page < total_pages,
            "first_row": start + 1 if total_rows else 0,
            "last_row": min(start + PAGE_SIZE, total_rows),
        },
    }


def parse_filters(
    query: Mapping[str, str], frame: pd.DataFrame
) -> tuple[AnalysisFilters, list[str]]:
    errors: list[str] = []
    age_min = _parse_optional_int(query.get("age_min"), "Minimalny wiek", errors)
    age_max = _parse_optional_int(query.get("age_max"), "Maksymalny wiek", errors)
    if age_min is not None and age_max is not None and age_min > age_max:
        errors.append("Minimalny wiek nie może być większy od maksymalnego.")

    sex = _parse_choice(query.get("sex"), set(frame["Sex"]), "płeć", errors)
    chest_pain = _parse_choice(
        query.get("chest_pain"), set(frame["ChestPainType"]), "typ bólu", errors
    )
    exercise_angina = _parse_choice(
        query.get("exercise_angina"),
        set(frame["ExerciseAngina"]),
        "dusznicę wysiłkową",
        errors,
    )
    heart_disease = _parse_optional_int(query.get("heart_disease"), "HeartDisease", errors)
    if heart_disease not in (None, 0, 1):
        errors.append("HeartDisease może mieć wyłącznie wartość 0 albo 1.")
        heart_disease = None

    return (
        AnalysisFilters(
            age_min=age_min,
            age_max=age_max,
            sex=sex,
            chest_pain=chest_pain,
            exercise_angina=exercise_angina,
            heart_disease=heart_disease,
        ),
        errors,
    )


def apply_filters(frame: pd.DataFrame, filters: AnalysisFilters) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    if filters.age_min is not None:
        mask &= frame["Age"] >= filters.age_min
    if filters.age_max is not None:
        mask &= frame["Age"] <= filters.age_max
    if filters.sex is not None:
        mask &= frame["Sex"] == filters.sex
    if filters.chest_pain is not None:
        mask &= frame["ChestPainType"] == filters.chest_pain
    if filters.exercise_angina is not None:
        mask &= frame["ExerciseAngina"] == filters.exercise_angina
    if filters.heart_disease is not None:
        mask &= frame[TARGET_COLUMN] == filters.heart_disease
    return frame.loc[mask].copy()


def _parse_optional_int(value: str | None, label: str, errors: list[str]) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        errors.append(f"Pole „{label}” musi być liczbą całkowitą.")
        return None


def _parse_choice(
    value: str | None,
    allowed: set[str],
    label: str,
    errors: list[str],
) -> str | None:
    if value in (None, ""):
        return None
    if value not in allowed:
        errors.append(f"Wybrano nieprawidłową wartość dla pola „{label}”.")
        return None
    return value


def _optional_mean(series: pd.Series) -> float | None:
    return float(series.mean()) if not series.empty else None


def _age_max_hr_scatter(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return _empty_figure("Wiek i maksymalne tętno")
    plot_frame = frame.assign(
        status=frame[TARGET_COLUMN].map({0: "Klasa 0", 1: "Klasa 1"})
    )
    figure = px.scatter(
        plot_frame,
        x="Age",
        y="MaxHR",
        color="status",
        symbol="Sex",
        opacity=0.72,
        color_discrete_map={"Klasa 0": COLOR_HEALTHY, "Klasa 1": COLOR_DISEASE},
        hover_data={"ChestPainType": True, "ExerciseAngina": True, "status": False},
        labels={
            "Age": "Wiek",
            "MaxHR": "Maksymalne tętno",
            "status": "HeartDisease",
            "Sex": "Płeć",
        },
    )
    return _style_figure(figure, "Wiek i maksymalne tętno")


def _target_distribution(frame: pd.DataFrame) -> go.Figure:
    counts = frame[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    figure = go.Figure(
        go.Bar(
            x=["Klasa 0", "Klasa 1"],
            y=counts.tolist(),
            marker_color=[COLOR_HEALTHY, COLOR_DISEASE],
            text=counts.tolist(),
            textposition="outside",
            hovertemplate="%{x}<br>Rekordy: %{y}<extra></extra>",
        )
    )
    return _style_figure(figure, "Klasy po zastosowaniu filtrów")


def _cholesterol_distribution(frame: pd.DataFrame) -> go.Figure:
    measured = frame.loc[frame["Cholesterol"] > 0].copy()
    if measured.empty:
        return _empty_figure("Rozkład zmierzonego cholesterolu")
    measured["status"] = measured[TARGET_COLUMN].map({0: "Klasa 0", 1: "Klasa 1"})
    figure = px.histogram(
        measured,
        x="Cholesterol",
        color="status",
        nbins=22,
        barmode="overlay",
        opacity=0.72,
        color_discrete_map={"Klasa 0": COLOR_HEALTHY, "Klasa 1": COLOR_DISEASE},
        labels={"Cholesterol": "Cholesterol [mg/dl]", "status": "HeartDisease"},
    )
    return _style_figure(figure, "Rozkład zmierzonego cholesterolu")


def _empty_figure(title: str) -> go.Figure:
    figure = go.Figure()
    figure.add_annotation(
        text="Brak rekordów dla wybranych filtrów",
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 15, "color": "#607885"},
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return _style_figure(figure, title)


def _style_figure(figure: go.Figure, title: str) -> go.Figure:
    figure.update_layout(
        title={"text": title, "font": {"size": 18, "color": "#152a3a"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif", "color": "#38505f"},
        margin={"l": 48, "r": 20, "t": 62, "b": 44},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hoverlabel={"bgcolor": "#152a3a", "font_color": "white"},
    )
    figure.update_xaxes(gridcolor="#e9eff2", linecolor="#dbe4e8")
    figure.update_yaxes(gridcolor="#e9eff2", zeroline=False)
    return figure


def _chart_html(figure: go.Figure) -> str:
    return to_html(
        figure,
        full_html=False,
        include_plotlyjs=False,
        config={"displaylogo": False, "responsive": True},
        default_height="370px",
    )

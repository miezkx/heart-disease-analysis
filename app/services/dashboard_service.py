from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.io import to_html

from cardio_ml.data import FEATURE_COLUMNS, TARGET_COLUMN

PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}
COLOR_HEALTHY = "#2f6b84"
COLOR_DISEASE = "#db5f57"


def build_dashboard_context(frame: pd.DataFrame) -> dict[str, Any]:
    positive_count = int(frame[TARGET_COLUMN].sum())
    row_count = int(len(frame))
    positive_rate = positive_count / row_count if row_count else 0.0

    return {
        "summary": {
            "rows": row_count,
            "features": len(FEATURE_COLUMNS),
            "positive_count": positive_count,
            "positive_rate": positive_rate,
            "mean_age": float(frame["Age"].mean()),
        },
        "charts": {
            "target": _chart_html(_target_distribution(frame)),
            "age": _chart_html(_age_distribution(frame)),
            "chest_pain": _chart_html(_chest_pain_distribution(frame)),
            "max_hr": _chart_html(_max_hr_distribution(frame)),
        },
    }


def _target_distribution(frame: pd.DataFrame) -> go.Figure:
    counts = frame[TARGET_COLUMN].value_counts().reindex([0, 1], fill_value=0)
    figure = go.Figure(
        data=[
            go.Bar(
                x=["Brak etykiety choroby", "Etykieta choroby"],
                y=counts.tolist(),
                marker_color=[COLOR_HEALTHY, COLOR_DISEASE],
                text=counts.tolist(),
                textposition="outside",
                hovertemplate="%{x}<br>Rekordy: %{y}<extra></extra>",
            )
        ]
    )
    return _style_figure(figure, "Rozkład zmiennej docelowej", "Liczba rekordów")


def _age_distribution(frame: pd.DataFrame) -> go.Figure:
    plot_frame = frame.assign(
        status=frame[TARGET_COLUMN].map({0: "Klasa 0", 1: "Klasa 1"})
    )
    figure = px.histogram(
        plot_frame,
        x="Age",
        color="status",
        barmode="overlay",
        opacity=0.72,
        nbins=20,
        color_discrete_map={"Klasa 0": COLOR_HEALTHY, "Klasa 1": COLOR_DISEASE},
        labels={"Age": "Wiek", "count": "Liczba rekordów", "status": "HeartDisease"},
    )
    return _style_figure(figure, "Wiek a etykieta HeartDisease", "Liczba rekordów")


def _chest_pain_distribution(frame: pd.DataFrame) -> go.Figure:
    grouped = (
        frame.groupby(["ChestPainType", TARGET_COLUMN], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    grouped["status"] = grouped[TARGET_COLUMN].map({0: "Klasa 0", 1: "Klasa 1"})
    figure = px.bar(
        grouped,
        x="ChestPainType",
        y="count",
        color="status",
        barmode="group",
        color_discrete_map={"Klasa 0": COLOR_HEALTHY, "Klasa 1": COLOR_DISEASE},
        labels={
            "ChestPainType": "Typ bólu w klatce piersiowej",
            "count": "Liczba rekordów",
            "status": "HeartDisease",
        },
    )
    return _style_figure(figure, "Typ bólu w klatce piersiowej", "Liczba rekordów")


def _max_hr_distribution(frame: pd.DataFrame) -> go.Figure:
    plot_frame = frame.assign(
        status=frame[TARGET_COLUMN].map({0: "Klasa 0", 1: "Klasa 1"})
    )
    figure = px.box(
        plot_frame,
        x="status",
        y="MaxHR",
        color="status",
        points="outliers",
        color_discrete_map={"Klasa 0": COLOR_HEALTHY, "Klasa 1": COLOR_DISEASE},
        labels={"status": "HeartDisease", "MaxHR": "Maksymalne tętno"},
    )
    figure.update_layout(showlegend=False)
    return _style_figure(figure, "Maksymalne tętno w obu klasach", "Uderzenia/min")


def _style_figure(figure: go.Figure, title: str, y_axis_title: str) -> go.Figure:
    figure.update_layout(
        title={"text": title, "font": {"size": 18, "color": "#152a3a"}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif", "color": "#38505f"},
        margin={"l": 46, "r": 20, "t": 56, "b": 42},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hoverlabel={"bgcolor": "#152a3a", "font_color": "white"},
    )
    figure.update_xaxes(showgrid=False, linecolor="#dbe4e8")
    figure.update_yaxes(title=y_axis_title, gridcolor="#e9eff2", zeroline=False)
    return figure


def _chart_html(figure: go.Figure) -> str:
    return to_html(
        figure,
        full_html=False,
        include_plotlyjs=False,
        config=PLOT_CONFIG,
        default_height="360px",
    )

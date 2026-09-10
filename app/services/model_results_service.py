import json
from datetime import datetime
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from plotly.io import to_html

MODEL_COLORS = {
    "logistic_regression": "#2f6b84",
    "decision_tree": "#d69b3c",
    "random_forest": "#0d7b78",
}
REQUIRED_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")


def load_model_results(
    results_path: str | Path,
    metadata_path: str | Path,
) -> dict[str, Any]:
    results = _read_json(Path(results_path))
    metadata = _read_json(Path(metadata_path))
    selected_key = results.get("selected_model")
    models = results.get("models", {})
    if selected_key not in models:
        raise ValueError("Plik wyników nie wskazuje prawidłowego wybranego modelu.")

    rows = []
    confusion_charts = []
    for model_key, model in models.items():
        _validate_model_result(model_key, model)
        cv = model["cross_validation"]
        test = model["test"]
        rows.append(
            {
                "key": model_key,
                "display_name": model["display_name"],
                "selected": model_key == selected_key,
                "cv": cv,
                "test": test["metrics"],
                "parameters": [
                    {
                        "name": name.removeprefix("model__"),
                        "value": "None" if value is None else str(value),
                    }
                    for name, value in model["best_parameters"].items()
                ],
            }
        )
        confusion_charts.append(
            {
                "display_name": model["display_name"],
                "selected": model_key == selected_key,
                "html": _chart_html(
                    _confusion_matrix_figure(
                        model["display_name"],
                        test["confusion_matrix"],
                        MODEL_COLORS.get(model_key, "#0d7b78"),
                    )
                ),
            }
        )

    selected = next(row for row in rows if row["selected"])
    return {
        "models": rows,
        "selected": selected,
        "selection_policy": results["selection_policy"],
        "dataset": results["dataset"],
        "metadata": {
            **metadata,
            "trained_at_display": _format_timestamp(metadata.get("trained_at")),
        },
        "roc_chart": _chart_html(_roc_figure(models)),
        "confusion_charts": confusion_charts,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Brak pliku {path.name}. Uruchom: python -m scripts.train_models"
        )
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Plik {path.name} nie zawiera prawidłowego JSON.") from error
    if not isinstance(content, dict):
        raise ValueError(f"Nieprawidłowa struktura pliku {path.name}.")
    return content


def _validate_model_result(model_key: str, model: dict[str, Any]) -> None:
    try:
        cv = model["cross_validation"]
        test = model["test"]
        for metric in REQUIRED_METRICS:
            cv[metric]["mean"]
            cv[metric]["std"]
            test["metrics"][metric]
        matrix = test["confusion_matrix"]
        if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
            raise KeyError
        test["roc_curve"]["false_positive_rate"]
        test["roc_curve"]["true_positive_rate"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Niepełne wyniki modelu: {model_key}.") from error


def _roc_figure(models: dict[str, dict[str, Any]]) -> go.Figure:
    figure = go.Figure()
    for model_key, model in models.items():
        roc = model["test"]["roc_curve"]
        auc = model["test"]["metrics"]["roc_auc"]
        figure.add_trace(
            go.Scatter(
                x=roc["false_positive_rate"],
                y=roc["true_positive_rate"],
                mode="lines",
                name=f"{model['display_name']} · AUC {auc:.3f}",
                line={"width": 3, "color": MODEL_COLORS.get(model_key)},
                hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Klasyfikator losowy",
            line={"color": "#9cafb7", "dash": "dash", "width": 2},
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        title={"text": "Krzywe ROC na zbiorze testowym", "font": {"size": 19}},
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1.02]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return _style_figure(figure, height=480)


def _confusion_matrix_figure(name: str, matrix: list[list[int]], color: str) -> go.Figure:
    maximum = max(max(row) for row in matrix) or 1
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predykcja 0", "Predykcja 1"],
            y=["Rzeczywista 0", "Rzeczywista 1"],
            colorscale=[[0, "#f3f7f8"], [1, color]],
            zmin=0,
            zmax=maximum,
            showscale=False,
            text=matrix,
            texttemplate="%{text}",
            textfont={"size": 19},
            hovertemplate="%{y}<br>%{x}<br>Rekordy: %{z}<extra></extra>",
        )
    )
    figure.update_layout(title={"text": name, "font": {"size": 17}})
    return _style_figure(figure, height=330)


def _style_figure(figure: go.Figure, height: int) -> go.Figure:
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif", "color": "#38505f"},
        margin={"l": 58, "r": 20, "t": 76, "b": 52},
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
    )


def _format_timestamp(value: str | None) -> str:
    if not value:
        return "brak informacji"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return parsed.strftime("%d.%m.%Y, %H:%M UTC")

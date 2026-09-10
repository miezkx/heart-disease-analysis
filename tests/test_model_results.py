import json

from app.services.model_results_service import load_model_results


def model_payload(name: str, auc: float) -> dict:
    metrics = {
        "accuracy": 0.8,
        "precision": 0.81,
        "recall": 0.82,
        "f1": 0.815,
        "roc_auc": auc,
    }
    return {
        "display_name": name,
        "best_parameters": {"model__max_depth": 3},
        "cross_validation": {
            key: {"mean": value, "std": 0.02} for key, value in metrics.items()
        },
        "test": {
            "metrics": metrics,
            "confusion_matrix": [[8, 2], [1, 9]],
            "roc_curve": {
                "false_positive_rate": [0.0, 0.2, 1.0],
                "true_positive_rate": [0.0, 0.9, 1.0],
                "thresholds": [None, 0.5, 0.0],
            },
        },
    }


def write_result_files(tmp_path):
    results_path = tmp_path / "evaluation_results.json"
    metadata_path = tmp_path / "model_metadata.json"
    results = {
        "selected_model": "random_forest",
        "selection_policy": {
            "primary_metric": "roc_auc",
            "test_size": 0.2,
            "cv_folds": 5,
            "random_state": 42,
            "decision_threshold": 0.5,
        },
        "dataset": {"rows": 100, "sha256": "abc"},
        "models": {
            "logistic_regression": model_payload("Regresja logistyczna", 0.88),
            "decision_tree": model_payload("Drzewo decyzyjne", 0.84),
            "random_forest": model_payload("Random Forest", 0.9),
        },
    }
    metadata = {
        "trained_at": "2026-09-09T12:00:00+00:00",
        "scikit_learn_version": "1.9.0",
    }
    results_path.write_text(json.dumps(results), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    return results_path, metadata_path


def test_load_model_results_marks_selected_model(tmp_path):
    results_path, metadata_path = write_result_files(tmp_path)

    context = load_model_results(results_path, metadata_path)

    assert context["selected"]["display_name"] == "Random Forest"
    assert sum(model["selected"] for model in context["models"]) == 1
    assert "Plotly.newPlot" in context["roc_chart"]


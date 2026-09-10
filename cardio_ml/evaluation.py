from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def evaluate_classifier(model: Any, features: Any, target: Any) -> dict[str, Any]:
    probability = model.predict_proba(features)[:, 1]
    prediction = (probability >= 0.5).astype(int)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(target, probability)

    return {
        "metrics": {
            "accuracy": float(accuracy_score(target, prediction)),
            "precision": float(precision_score(target, prediction, zero_division=0)),
            "recall": float(recall_score(target, prediction, zero_division=0)),
            "f1": float(f1_score(target, prediction, zero_division=0)),
            "roc_auc": float(roc_auc_score(target, probability)),
        },
        "confusion_matrix": confusion_matrix(target, prediction, labels=[0, 1]).tolist(),
        "roc_curve": {
            "false_positive_rate": false_positive_rate.tolist(),
            "true_positive_rate": true_positive_rate.tolist(),
            "thresholds": _finite_list(thresholds),
        },
    }


def _finite_list(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]

import pandas as pd
from sklearn.linear_model import LogisticRegression

from cardio_ml.evaluation import evaluate_classifier


def test_evaluation_contains_required_metrics():
    features = pd.DataFrame({"feature": [0, 1, 2, 3, 4, 5]})
    target = pd.Series([0, 0, 0, 1, 1, 1])
    model = LogisticRegression().fit(features, target)

    result = evaluate_classifier(model, features, target)

    assert set(result["metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert len(result["confusion_matrix"]) == 2
    assert len(result["roc_curve"]["false_positive_rate"]) > 1


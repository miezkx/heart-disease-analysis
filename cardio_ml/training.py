import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from cardio_ml.data import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset
from cardio_ml.evaluation import evaluate_classifier
from cardio_ml.pipelines import RANDOM_STATE, build_model_specs, build_pipeline

PRIMARY_METRIC = "roc_auc"
TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING = ("accuracy", "precision", "recall", "f1", "roc_auc")


def train_and_select(
    dataset_path: str | Path,
    artifact_directory: str | Path = "artifacts",
) -> dict[str, Any]:
    source_path = Path(dataset_path)
    artifact_path = Path(artifact_directory)
    artifact_path.mkdir(parents=True, exist_ok=True)

    frame = load_dataset(source_path)
    target_values = set(frame[TARGET_COLUMN].unique().tolist())
    if not target_values <= {0, 1}:
        raise ValueError(f"HeartDisease musi zawierać tylko 0 i 1: {target_values}")

    features = frame.loc[:, FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )
    cross_validation = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fitted_searches: dict[str, GridSearchCV] = {}
    comparison: dict[str, Any] = {}

    for model_key, spec in build_model_specs().items():
        search = GridSearchCV(
            estimator=build_pipeline(spec.estimator),
            param_grid=spec.parameter_grid,
            scoring=list(SCORING),
            refit=PRIMARY_METRIC,
            cv=cross_validation,
            n_jobs=-1,
            return_train_score=False,
        )
        search.fit(train_features, train_target)
        fitted_searches[model_key] = search
        comparison[model_key] = {
            "display_name": spec.display_name,
            "best_parameters": _json_ready(search.best_params_),
            "cross_validation": _best_cv_scores(search),
            "test": evaluate_classifier(search.best_estimator_, test_features, test_target),
        }

    selected_key = max(
        fitted_searches,
        key=lambda key: fitted_searches[key].best_score_,
    )
    selected_search = fitted_searches[selected_key]

    deployment_pipeline = clone(selected_search.best_estimator_)
    deployment_pipeline.fit(features, target)
    joblib.dump(deployment_pipeline, artifact_path / "best_pipeline.joblib")

    generated_at = datetime.now(UTC).isoformat()
    results = {
        "generated_at": generated_at,
        "selection_policy": {
            "primary_metric": PRIMARY_METRIC,
            "source": "mean cross-validation score on the training partition",
            "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS,
            "random_state": RANDOM_STATE,
            "decision_threshold": 0.5,
        },
        "dataset": {
            "rows": int(len(frame)),
            "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        },
        "selected_model": selected_key,
        "models": comparison,
    }
    metadata = {
        "model_version": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "trained_at": generated_at,
        "selected_model": selected_key,
        "best_parameters": _json_ready(selected_search.best_params_),
        "feature_columns": list(FEATURE_COLUMNS),
        "target_column": TARGET_COLUMN,
        "decision_threshold": 0.5,
        "dataset_sha256": results["dataset"]["sha256"],
        "scikit_learn_version": sklearn.__version__,
        "disclaimer": "Model edukacyjny; wynik nie stanowi diagnozy medycznej.",
    }

    _write_json(artifact_path / "evaluation_results.json", results)
    _write_json(artifact_path / "model_metadata.json", metadata)
    return results


def _best_cv_scores(search: GridSearchCV) -> dict[str, dict[str, float]]:
    index = search.best_index_
    return {
        metric: {
            "mean": float(search.cv_results_[f"mean_test_{metric}"][index]),
            "std": float(search.cv_results_[f"std_test_{metric}"][index]),
        }
        for metric in SCORING
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")

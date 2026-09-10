from dataclasses import dataclass
from typing import Any

from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from cardio_ml.preprocessing import build_preprocessor

RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    estimator: ClassifierMixin
    parameter_grid: dict[str, list[Any]]


def build_model_specs() -> dict[str, ModelSpec]:
    return {
        "logistic_regression": ModelSpec(
            display_name="Regresja logistyczna",
            estimator=LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            parameter_grid={
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "decision_tree": ModelSpec(
            display_name="Drzewo decyzyjne",
            estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
            parameter_grid={
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [1, 5, 10],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": ModelSpec(
            display_name="Random Forest",
            estimator=RandomForestClassifier(
                n_estimators=300,
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            parameter_grid={
                "model__max_depth": [None, 6, 10],
                "model__min_samples_leaf": [1, 3, 6],
                "model__class_weight": [None, "balanced"],
            },
        ),
    }


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline(steps=[("preprocessor", build_preprocessor()), ("model", estimator)])

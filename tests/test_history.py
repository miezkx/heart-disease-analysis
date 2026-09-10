from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.exc import IntegrityError

from app import create_app
from app.extensions import db
from app.models import PredictionRecord
from app.repositories.prediction_repository import prediction_summary


def history_app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": (
                f"sqlite:///{(tmp_path / 'history-tests.db').as_posix()}"
            ),
        }
    )


def make_record(index: int = 1, probability: float = 0.25) -> PredictionRecord:
    return PredictionRecord(
        created_at=datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=index),
        predicted_class=int(probability >= 0.5),
        probability=probability,
        threshold=0.5,
        model_name="Random Forest",
        model_version=f"wersja-{index}",
        input_data={"Age": 40 + index, "Sex": "M"},
        explanation_data=[],
    )


def test_empty_history_has_clear_next_action(tmp_path):
    response = history_app(tmp_path).test_client().get("/history")

    assert response.status_code == 200
    assert "Historia jest jeszcze pusta" in response.text
    assert "Przejdź do predykcji" in response.text


def test_history_is_paginated_from_newest_record(tmp_path):
    app = history_app(tmp_path)
    with app.app_context():
        db.session.add_all(make_record(index) for index in range(1, 18))
        db.session.commit()

    first_page = app.test_client().get("/history")
    second_page = app.test_client().get("/history?page=2")

    assert "<td>#17</td>" in first_page.text
    assert "<td>#1</td>" not in first_page.text
    assert "<td>#1</td>" in second_page.text
    assert "1–15 z 17" in first_page.text
    assert "16–17 z 17" in second_page.text


def test_prediction_summary_counts_class_one(tmp_path):
    app = history_app(tmp_path)
    with app.app_context():
        db.session.add_all(
            [make_record(1, 0.2), make_record(2, 0.7), make_record(3, 0.8)]
        )
        db.session.commit()

        summary = prediction_summary()

    assert summary == {
        "total": 3,
        "class_one": 2,
        "class_one_percent": pytest.approx(66.6667),
    }


def test_database_rejects_probability_outside_zero_one(tmp_path):
    app = history_app(tmp_path)
    with app.app_context(), pytest.raises(IntegrityError):
        db.session.add(make_record(probability=1.2))
        db.session.commit()

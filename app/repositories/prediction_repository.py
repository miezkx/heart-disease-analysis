from typing import Any

from flask_sqlalchemy.pagination import Pagination

from app.extensions import db
from app.models import PredictionRecord


def save_prediction(
    patient: dict[str, Any], result: dict[str, Any]
) -> PredictionRecord:
    record = PredictionRecord(
        predicted_class=int(result["predicted_class"]),
        probability=float(result["probability"]),
        threshold=float(result["threshold"]),
        model_name=str(result["model_name"]),
        model_version=str(result["model_version"]),
        input_data=patient,
        explanation_data=result["explanations"],
    )
    db.session.add(record)
    db.session.commit()
    return record


def prediction_history(page: int, per_page: int = 15) -> Pagination:
    statement = db.select(PredictionRecord).order_by(
        PredictionRecord.created_at.desc(), PredictionRecord.id.desc()
    )
    return db.paginate(statement, page=page, per_page=per_page, error_out=False)


def prediction_summary() -> dict[str, int | float]:
    total = db.session.scalar(db.select(db.func.count()).select_from(PredictionRecord)) or 0
    class_one = (
        db.session.scalar(
            db.select(db.func.count())
            .select_from(PredictionRecord)
            .where(PredictionRecord.predicted_class == 1)
        )
        or 0
    )
    return {
        "total": total,
        "class_one": class_one,
        "class_one_percent": class_one / total * 100 if total else 0.0,
    }

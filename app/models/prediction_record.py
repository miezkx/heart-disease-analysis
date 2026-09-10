from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, CheckConstraint, DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.extensions import db


class PredictionRecord(db.Model):
    __tablename__ = "prediction_records"
    __table_args__ = (
        CheckConstraint("predicted_class IN (0, 1)", name="ck_prediction_class"),
        CheckConstraint(
            "probability >= 0 AND probability <= 1", name="ck_prediction_probability"
        ),
        CheckConstraint("threshold >= 0 AND threshold <= 1", name="ck_prediction_threshold"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC), index=True
    )
    predicted_class: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold: Mapped[float] = mapped_column(Float, nullable=False)
    model_name: Mapped[str] = mapped_column(String(80), nullable=False)
    model_version: Mapped[str] = mapped_column(String(80), nullable=False)
    input_data: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    explanation_data: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)

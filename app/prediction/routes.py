from pathlib import Path

from flask import Blueprint, current_app, render_template, request
from sqlalchemy.exc import SQLAlchemyError

from app.extensions import db
from app.prediction.forms import parse_patient_input
from app.repositories.prediction_repository import save_prediction
from app.services.prediction_service import predict_patient
from cardio_ml.data import load_dataset

blueprint = Blueprint("prediction", __name__, url_prefix="/prediction")


@blueprint.route("", methods=["GET", "POST"])
def index() -> tuple[str, int] | str:
    values = request.form.to_dict() if request.method == "POST" else {}
    errors: dict[str, str] = {}
    result = None
    status = 200

    if request.method == "POST":
        patient, errors = parse_patient_input(request.form)
        if errors:
            status = 422
        elif patient is not None:
            try:
                result = predict_patient(
                    patient,
                    load_dataset(current_app.config["DATASET_PATH"]),
                    current_app.config["MODEL_ARTIFACT_PATH"],
                    current_app.config["MODEL_METADATA_PATH"],
                )
                try:
                    record = save_prediction(patient, result)
                    result["history_id"] = record.id
                except SQLAlchemyError:
                    db.session.rollback()
                    errors["history"] = (
                        "Wynik obliczono, ale nie udało się zapisać go w historii."
                    )
            except (FileNotFoundError, ValueError) as error:
                errors["model"] = _model_error_message(error)
                status = 503

    rendered = render_template(
        "prediction.html",
        active_page="prediction",
        values=values,
        errors=errors,
        result=result,
    )
    return (rendered, status) if status != 200 else rendered


def _model_error_message(error: Exception) -> str:
    missing_name = Path(error.filename).name if isinstance(error, FileNotFoundError) else None
    detail = f" Brak pliku: {missing_name}." if missing_name else ""
    return f"Model nie jest dostępny.{detail} Uruchom: python -m scripts.train_models"

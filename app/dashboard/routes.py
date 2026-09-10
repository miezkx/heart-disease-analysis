from flask import Blueprint, current_app, render_template

from app.services.dashboard_service import build_dashboard_context
from cardio_ml.data import audit_dataset, load_dataset

blueprint = Blueprint("dashboard", __name__)


@blueprint.get("/")
def index() -> str:
    frame = load_dataset(current_app.config["DATASET_PATH"])
    return render_template(
        "dashboard.html",
        **build_dashboard_context(frame),
        audit=audit_dataset(frame),
        active_page="dashboard",
    )

from flask import Blueprint, render_template, request

from app.prediction.forms import FIELD_LABELS
from app.repositories.prediction_repository import prediction_history, prediction_summary

blueprint = Blueprint("history", __name__, url_prefix="/history")


@blueprint.get("")
def index() -> str:
    page = max(request.args.get("page", default=1, type=int) or 1, 1)
    pagination = prediction_history(page)
    return render_template(
        "history.html",
        active_page="history",
        pagination=pagination,
        summary=prediction_summary(),
        field_labels=FIELD_LABELS,
    )

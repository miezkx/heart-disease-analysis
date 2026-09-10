from flask import Blueprint, current_app, render_template, request, url_for

from app.services.analysis_service import build_analysis_context
from cardio_ml.data import load_dataset

blueprint = Blueprint("analysis", __name__, url_prefix="/analysis")


@blueprint.get("")
def index() -> str:
    frame = load_dataset(current_app.config["DATASET_PATH"])
    requested_page = request.args.get("page", default=1, type=int) or 1
    context = build_analysis_context(frame, request.args, requested_page=requested_page)

    query = context["filters"].query_parameters()
    pagination = context["pagination"]
    pagination["previous_url"] = (
        url_for("analysis.index", **query, page=pagination["page"] - 1)
        if pagination["has_previous"]
        else None
    )
    pagination["next_url"] = (
        url_for("analysis.index", **query, page=pagination["page"] + 1)
        if pagination["has_next"]
        else None
    )
    return render_template("analysis.html", **context, active_page="analysis")

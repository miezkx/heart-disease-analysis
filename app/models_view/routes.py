from flask import Blueprint, current_app, render_template

from app.services.model_results_service import load_model_results

blueprint = Blueprint("models_view", __name__, url_prefix="/models")


@blueprint.get("")
def index() -> tuple[str, int] | str:
    try:
        context = load_model_results(
            current_app.config["EVALUATION_RESULTS_PATH"],
            current_app.config["MODEL_METADATA_PATH"],
        )
    except (FileNotFoundError, ValueError) as error:
        return (
            render_template(
                "models_unavailable.html",
                active_page="models",
                error_message=str(error),
            ),
            503,
        )
    return render_template("models.html", **context, active_page="models")

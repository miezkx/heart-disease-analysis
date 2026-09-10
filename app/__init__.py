from importlib import import_module
from pathlib import Path

import click
from flask import Flask

from app.config import Config
from app.extensions import db


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    db.init_app(app)

    import_module("app.models")
    from app.analysis.routes import blueprint as analysis_blueprint
    from app.dashboard.routes import blueprint as dashboard_blueprint
    from app.history.routes import blueprint as history_blueprint
    from app.models_view.routes import blueprint as models_blueprint
    from app.prediction.routes import blueprint as prediction_blueprint

    app.register_blueprint(dashboard_blueprint)
    app.register_blueprint(analysis_blueprint)
    app.register_blueprint(models_blueprint)
    app.register_blueprint(prediction_blueprint)
    app.register_blueprint(history_blueprint)

    with app.app_context():
        db.create_all()

    @app.cli.command("init-db")
    def init_db_command() -> None:
        db.create_all()
        click.echo("Baza danych jest gotowa.")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app

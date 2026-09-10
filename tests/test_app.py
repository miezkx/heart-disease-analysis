import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy.exc import SQLAlchemyError

from app import create_app
from app.extensions import db
from app.models import PredictionRecord
from cardio_ml.data import FEATURE_COLUMNS
from cardio_ml.pipelines import build_pipeline
from tests.test_model_results import write_result_files
from tests.test_prediction import valid_form


def dashboard_rows() -> list[dict[str, object]]:
    base = {
        "Age": 40,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 140,
        "Cholesterol": 289,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 172,
        "ExerciseAngina": "N",
        "Oldpeak": 0.0,
        "ST_Slope": "Up",
    }
    return [{**base, "HeartDisease": 0}, {**base, "Age": 60, "HeartDisease": 1}]


def test_health_endpoint():
    client = create_app().test_client()

    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_dashboard_renders_dataset_summary(tmp_path):
    dataset_path = tmp_path / "heart.csv"
    pd.DataFrame(dashboard_rows()).to_csv(dataset_path, index=False)
    client = create_app({"TESTING": True, "DATASET_PATH": dataset_path}).test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert "Przegląd zbioru HeartDisease" in response.text
    assert "Rekordy" in response.text
    assert "Plotly.newPlot" in response.text


def test_analysis_route_filters_rows(tmp_path):
    dataset_path = tmp_path / "heart.csv"
    pd.DataFrame(dashboard_rows()).to_csv(dataset_path, index=False)
    client = create_app({"TESTING": True, "DATASET_PATH": dataset_path}).test_client()

    response = client.get("/analysis?heart_disease=1")

    assert response.status_code == 200
    assert "Filtrowanie i analiza" in response.text
    assert "Rekordy po filtrowaniu" in response.text
    assert "1–1 z 1" not in response.text


def test_models_route_renders_comparison(tmp_path):
    results_path, metadata_path = write_result_files(tmp_path)
    client = create_app(
        {
            "TESTING": True,
            "EVALUATION_RESULTS_PATH": results_path,
            "MODEL_METADATA_PATH": metadata_path,
        }
    ).test_client()

    response = client.get("/models")

    assert response.status_code == 200
    assert "Porównanie modeli" in response.text
    assert "Random Forest" in response.text
    assert response.text.index("cdn.plot.ly") < response.text.index("Plotly.newPlot")


def test_models_route_explains_missing_results(tmp_path):
    client = create_app(
        {
            "TESTING": True,
            "EVALUATION_RESULTS_PATH": tmp_path / "missing-results.json",
            "MODEL_METADATA_PATH": tmp_path / "missing-metadata.json",
        }
    ).test_client()

    response = client.get("/models")

    assert response.status_code == 503
    assert "python -m scripts.train_models" in response.text


def test_prediction_route_returns_model_result(tmp_path):
    rows = dashboard_rows() * 4
    for index, row in enumerate(rows):
        row["Age"] = 35 + index * 5
        row["HeartDisease"] = index % 2
    dataset_path = tmp_path / "heart.csv"
    frame = pd.DataFrame(rows)
    frame.to_csv(dataset_path, index=False)

    model = build_pipeline(RandomForestClassifier(n_estimators=10, random_state=42))
    model.fit(frame.loc[:, FEATURE_COLUMNS], frame["HeartDisease"])
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        '{"decision_threshold": 0.5, "selected_model": "random_forest", "model_version": "test"}',
        encoding="utf-8",
    )
    client = create_app(
        {
            "TESTING": True,
            "DATASET_PATH": dataset_path,
            "MODEL_ARTIFACT_PATH": model_path,
            "MODEL_METADATA_PATH": metadata_path,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{(tmp_path / 'test.db').as_posix()}",
        }
    ).test_client()

    response = client.post(
        "/prediction",
        data={
            "Age": "54",
            "Sex": "M",
            "ChestPainType": "ASY",
            "RestingBP": "140",
            "Cholesterol": "250",
            "FastingBS": "0",
            "RestingECG": "Normal",
            "MaxHR": "130",
            "ExerciseAngina": "Y",
            "Oldpeak": "1.5",
            "ST_Slope": "Flat",
        },
    )

    assert response.status_code == 200
    assert "Klasa " in response.text
    assert "prawdopodobieństwo modelowe klasy 1" not in response.text
    assert "Zapisano w historii jako rekord" not in response.text

    with client.application.app_context():
        record = db.session.execute(db.select(PredictionRecord)).scalar_one()
        assert record.predicted_class in {0, 1}
        assert record.input_data["Age"] == 54
        assert record.model_name == "Random Forest"


def test_history_route_lists_saved_prediction(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{(tmp_path / 'history.db').as_posix()}",
        }
    )
    with app.app_context():
        db.session.add(
            PredictionRecord(
                predicted_class=1,
                probability=0.82,
                threshold=0.5,
                model_name="Random Forest",
                model_version="test-v1",
                input_data={"Age": 61, "Sex": "M"},
                explanation_data=[],
            )
        )
        db.session.commit()

    response = app.test_client().get("/history")

    assert response.status_code == 200
    assert "Historia predykcji" in response.text
    assert "82.0%" in response.text
    assert "Random Forest" in response.text


def test_invalid_prediction_is_not_saved(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{(tmp_path / 'invalid.db').as_posix()}",
        }
    )

    response = app.test_client().post("/prediction", data={"Age": "100"})

    assert response.status_code == 422
    assert "Dozwolony zakres: 28–77" in response.text
    with app.app_context():
        count = db.session.scalar(db.select(db.func.count()).select_from(PredictionRecord))
    assert count == 0


def test_prediction_remains_visible_when_history_write_fails(tmp_path, monkeypatch):
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{(tmp_path / 'failure.db').as_posix()}",
        }
    )
    model_result = {
        "predicted_class": 1,
        "probability": 0.75,
        "probability_percent": 75.0,
        "threshold": 0.5,
        "model_name": "Random Forest",
        "model_version": "test",
        "explanations": [],
    }
    monkeypatch.setattr("app.prediction.routes.load_dataset", lambda path: pd.DataFrame())
    monkeypatch.setattr(
        "app.prediction.routes.predict_patient",
        lambda patient, frame, model_path, metadata_path: model_result,
    )

    def fail_to_save(patient, result):
        raise SQLAlchemyError("test database failure")

    monkeypatch.setattr("app.prediction.routes.save_prediction", fail_to_save)

    response = app.test_client().post("/prediction", data=valid_form())

    assert response.status_code == 200
    assert "Klasa 1" in response.text
    assert "nie udało się zapisać go w historii" in response.text

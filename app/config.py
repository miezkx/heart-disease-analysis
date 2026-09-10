from pathlib import Path

BASE_DIRECTORY = Path(__file__).resolve().parent.parent


class Config:
    DATASET_PATH = BASE_DIRECTORY / "data" / "raw" / "heart.csv"
    MODEL_ARTIFACT_PATH = BASE_DIRECTORY / "artifacts" / "best_pipeline.joblib"
    EVALUATION_RESULTS_PATH = BASE_DIRECTORY / "artifacts" / "evaluation_results.json"
    MODEL_METADATA_PATH = BASE_DIRECTORY / "artifacts" / "model_metadata.json"
    DATABASE_PATH = BASE_DIRECTORY / "instance" / "predictions.db"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH.as_posix()}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

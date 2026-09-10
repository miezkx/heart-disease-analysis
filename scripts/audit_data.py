import json
from pathlib import Path

from cardio_ml.data import audit_dataset, load_dataset

DEFAULT_DATASET_PATH = Path("data/raw/heart.csv")


def main() -> None:
    frame = load_dataset(DEFAULT_DATASET_PATH)
    report = audit_dataset(frame)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

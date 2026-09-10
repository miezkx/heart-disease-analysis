from cardio_ml.training import train_and_select


def main() -> None:
    results = train_and_select("data/raw/heart.csv")
    selected = results["selected_model"]
    print(f"Wybrany model: {results['models'][selected]['display_name']}")
    print("Wyniki zapisano w artifacts/evaluation_results.json")
    print("Pipeline zapisano w artifacts/best_pipeline.joblib")


if __name__ == "__main__":
    main()

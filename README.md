# Analiza i predykcja choroby serca

Projekt akademicki przedstawiający aplikację internetową do eksploracji danych
oraz porównania modeli klasyfikacyjnych przewidujących wartość zmiennej
`HeartDisease`.

Wykorzystany zbiór danych:
[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
Zawiera on 918 obserwacji i 11 cech wejściowych.

## Funkcjonalności

- charakterystyka zbioru i interaktywne wykresy,
- filtrowanie danych i przegląd rekordów,
- porównanie regresji logistycznej, drzewa decyzyjnego i Random Forest,
- metryki accuracy, precision, recall, F1 i ROC AUC,
- macierze pomyłek oraz krzywe ROC,
- formularz predykcji z prostym wyjaśnieniem wyniku,
- anonimowa historia predykcji zapisywana w SQLite.

## Technologie

Python 3.12, Flask, Pandas, scikit-learn, Plotly, SQLAlchemy, SQLite,
Bootstrap, joblib, pytest i Ruff.

## Uruchomienie

W terminalu PowerShell w katalogu projektu wykonaj:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
python run.py
```

Aplikacja będzie dostępna pod adresem <http://127.0.0.1:5000/>.

Plik `heart.csv` powinien znajdować się w katalogu `data/raw/`.

## Trenowanie modeli

```powershell
python -m scripts.train_models
```

Dane są dzielone w sposób stratyfikowany na część treningową i testową w
proporcji 80/20. Hiperparametry są dobierane przez `GridSearchCV` z
pięciokrotną walidacją krzyżową. Najlepszy pipeline jest zapisywany przez
joblib w katalogu `artifacts/`.

## Testy

```powershell
python -m pytest
python -m ruff check .
```

## Struktura projektu

```text
app/          aplikacja Flask, szablony i obsługa bazy danych
cardio_ml/    przygotowanie danych, modele, trening i ewaluacja
data/raw/     zbiór danych heart.csv
artifacts/    zapisany pipeline oraz wyniki modeli
instance/     lokalna baza SQLite
scripts/      skrypty audytu danych i trenowania modeli
tests/        testy automatyczne
docs/         dodatkowa dokumentacja projektu
run.py        punkt startowy aplikacji
```

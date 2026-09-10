# Audyt danych — etap 1

Źródło: `data/raw/heart.csv`, Heart Failure Prediction Dataset.

## Wyniki

- 918 rekordów i 12 kolumn,
- 11 cech wejściowych oraz etykieta `HeartDisease`,
- brak pustych komórek rozpoznawanych przez Pandas,
- brak identycznych, zduplikowanych wierszy,
- rozkład klas: 508 rekordów klasy `1` i 410 klasy `0`,
- 1 zerowa wartość `RestingBP`,
- 172 zerowe wartości `Cholesterol`.

## Wstępna decyzja preprocessingowa

Wartości `0` w `RestingBP` i `Cholesterol` będą traktowane jako brak pomiaru,
nie jako rzeczywista wartość fizjologiczna. Zamiana na `NaN` i imputacja medianą
muszą następować wewnątrz pipeline'u scikit-learn, po podziale danych. Zapobiega
to wykorzystaniu informacji ze zbioru walidacyjnego lub testowego.

`Oldpeak` zawiera wartości ujemne. Nie będą automatycznie usuwane ani zamieniane
bez dodatkowego uzasadnienia domenowego. W raporcie końcowym trzeba jawnie opisać
wszystkie przyjęte reguły jakości danych.

## Kolejny krok

Zbudowanie trzech porównywalnych pipeline'ów i ocena ich przez stratyfikowaną
walidację krzyżową, przed jednokrotnym użyciem wydzielonego zbioru testowego.


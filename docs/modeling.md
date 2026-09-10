# Metodyka i wyniki modelowania

## Cel

Zadaniem jest binarna klasyfikacja wartości `HeartDisease` na podstawie 11
cech. Klasa `1` oznacza dodatnią etykietę w datasecie, a nie diagnozę medyczną.

## Przygotowanie danych

Wspólny `ColumnTransformer` jest częścią każdego pipeline'u:

- wartości `0` w `RestingBP` i `Cholesterol` są traktowane jako brak danych,
- cechy numeryczne otrzymują imputację medianą i standaryzację,
- cechy kategoryczne otrzymują imputację dominantą i kodowanie One-Hot,
- wszystkie transformacje są dopasowywane wewnątrz walidacji krzyżowej.

Umieszczenie preprocessingu w pipeline ogranicza wyciek informacji z części
walidacyjnej i testowej.

## Eksperyment

- podział trening/test: 80/20 ze stratyfikacją,
- `random_state = 42`,
- walidacja: `StratifiedKFold`, 5 podziałów z tasowaniem,
- strojenie: `GridSearchCV`,
- główna metryka wyboru: średni ROC AUC z walidacji krzyżowej,
- próg przypisania klasy: 0,5.

Porównywane algorytmy to regresja logistyczna, drzewo decyzyjne oraz Random
Forest. Zbiór testowy nie uczestniczy w wyborze zwycięzcy.

## Wyniki bieżącego treningu

| Model | CV ROC AUC | Accuracy test | Precision test | Recall test | F1 test | ROC AUC test |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Regresja logistyczna | 0,9223 | 0,8859 | 0,8857 | 0,9118 | 0,8986 | 0,9329 |
| Drzewo decyzyjne | 0,9074 | 0,8207 | 0,8710 | 0,7941 | 0,8308 | 0,8890 |
| Random Forest | **0,9341** | **0,8913** | **0,8942** | **0,9118** | **0,9029** | 0,9305 |

Wybrano Random Forest, ponieważ uzyskał najwyższe średnie ROC AUC w walidacji
krzyżowej. Najlepsza konfiguracja używa `class_weight="balanced"`,
`min_samples_leaf=3` i nie ogranicza `max_depth`.

Metryki testowe opisują model dopasowany wyłącznie na części treningowej.
Artefakt używany przez aplikację został później ponownie dopasowany do całego
datasetu, dlatego nie należy utożsamiać jego działania na nowych danych z
wynikiem pojedynczego testowego podziału.

## Interpretacja metryk

- **Accuracy** — udział wszystkich poprawnych klasyfikacji.
- **Precision** — jaki odsetek rekordów przewidzianych jako klasa 1 rzeczywiście
  miał etykietę 1.
- **Recall** — jaki odsetek rekordów klasy 1 został odnaleziony.
- **F1** — średnia harmoniczna precision i recall.
- **ROC AUC** — zdolność modelu do porządkowania klas niezależnie od jednego
  wybranego progu.

## Wyjaśnienie pojedynczego wyniku

Dla każdej cechy aplikacja tworzy wariant rekordu, w którym tylko tę cechę
zastępuje wartością typową ze zbioru treningowego. Różnica prawdopodobieństwa
jest raportowana w punktach procentowych. Cztery największe bezwzględne zmiany
są prezentowane użytkownikowi.

Jest to prosta analiza wrażliwości. Nie dowodzi związku przyczynowego i nie jest
odpowiednikiem interpretacji klinicznej.

## Ograniczenia

- dataset jest niewielki i może nie reprezentować innych populacji,
- dane nie zawierają pełnego kontekstu klinicznego,
- wartości zerowe cholesterolu są liczne i wymagają imputacji,
- nie wykonano zewnętrznej walidacji na niezależnym źródle,
- prawdopodobieństwo modelowe nie zostało skalibrowane jako ryzyko kliniczne,
- lokalne wyjaśnienie bada wrażliwość, a nie przyczynowość.

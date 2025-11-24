# Rainfall Prediction in Australia — ML Pipeline

Projekt obejmuje pełny pipeline predykcji opadów w Australii, wykorzystując dane **weatherAUS**.
Zawiera preprocessing, feature engineering, strojenie modeli Optuna, klasyfikację (czy jutro będzie padać) oraz regresję (ile mm deszczu spadnie).

---

## Struktura projektu

1. **Przygotowanie danych**

   * Parsowanie daty, sortowanie po `Location` i `Date`
   * Czyszczenie wartości odstających (clamping)
   * Tworzenie nowych cech (różnice temperatur, wilgotności, ciśnienia, zakres temperatur, cechy sezonowe)
   * Lags dla wybranych zmiennych (1- i 2-dniowe)

2. **Podział na zbiory czasowe**

   * 70% train / 15% validation / 15% test
   * Chroni przed wyciekiem informacji czasowej

3. **Kodowanie zmiennych kategorycznych**

   * CatBoost: typ `category`
   * XGBoost/LightGBM: `OrdinalEncoder` z obsługą nieznanych wartości

4. **Modele klasyfikacyjne**

   * XGBoost, LightGBM, CatBoost
   * Optuna + TPESampler(seed=42)
   * F1-score jako funkcja celu
   * Ensemble: średnia prognoz trzech modeli + threshold tuning

5. **Regresja opadów (mm)**

   * Tylko dni z opadami (`RainTomorrow == 1`)
   * Log-transformacja celu (`log1p`)
   * XGBRegressor + Optuna
   * Ewaluacja: RMSE i MAE na zbiorze testowym

---

## Wyniki

* Ensemble klasyfikacyjny: F1-score, ROC AUC, classification report
* Regresja: RMSE i MAE w mm

---

## Wymagania

```
numpy
pandas
optuna
scikit-learn
xgboost
lightgbm
catboost
```


import numpy as np
import pandas as pd
import optuna
import warnings
from sklearn.preprocessing import OrdinalEncoder
from optuna.samplers import TPESampler

# Modele
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

# Metryki
from sklearn.metrics import (f1_score, roc_auc_score, accuracy_score, 
                             classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error)

# Early stoppingi
from lightgbm import early_stopping
from xgboost.callback import EarlyStopping

# Wyłączamy ostrzeżenia
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)


# 1. WCZYTANIE I CZYSZCZENIE DANYCH


df = pd.read_csv('weatherAUS.csv')

# Konwersja daty
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Sortowanie: Najpierw Lokalizacja, potem Data (kluczowe dla Lags)
df = df.sort_values(['Location', 'Date']).reset_index(drop=True)

# Usunięcie braków w zmiennej celu
df = df.dropna(subset=['RainTomorrow']).copy()
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

# Funkcja Clamp
def clamp(series, lo=None, hi=None):
    if lo is not None: series = series.clip(lower=lo)
    if hi is not None: series = series.clip(upper=hi)
    return series

for col in ['Humidity9am', 'Humidity3pm']: df[col] = clamp(df[col], 0, 100)
for col in ['Cloud9am', 'Cloud3pm']: df[col] = clamp(df[col], 0, 9)
nonneg_cols = ["Rainfall", "Evaporation", "Pressure9am", "Pressure3pm", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"]
for col in nonneg_cols: df[col] = clamp(df[col], 0, None)

# Cechy dodatkowe
df['RainToday_bin'] = df['RainToday'].map({'Yes': 1, 'No': 0})
df["temp_range"] = df["MaxTemp"] - df["MinTemp"]
df["temp_delta_9to3"] = df["Temp3pm"] - df["Temp9am"]
df["humidity_delta_9to3"] = df["Humidity3pm"] - df["Humidity9am"]
df["pressure_delta_9to3"] = df["Pressure3pm"] - df["Pressure9am"]
df["wind_delta_9to3"] = df["WindSpeed3pm"] - df["WindSpeed9am"]

dayofyear = df["Date"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.0)
df["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365.0)
df["Month"] = df["Date"].dt.month


# 2. FEATURE ENGINEERING: LAGS

cols_to_lag = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine',
    'WindGustSpeed', 'Humidity3pm', 'Pressure3pm', 'Cloud3pm', 'Temp3pm'
]

for col in cols_to_lag:
    df[f'{col}_lag1'] = df.groupby('Location')[col].shift(1)
    df[f'{col}_lag2'] = df.groupby('Location')[col].shift(2)


# 3. PRZYGOTOWANIE ZBIORÓW 

# Sortowanie chronologiczne (Zostało przeniesione z poprzedniego bloku, bo jest kluczowe)
df = df.sort_values('Date').reset_index(drop=True)

# Definicja masek czasowych
dates_sorted = df['Date']
t1 = dates_sorted.iloc[int(len(dates_sorted) * 0.7)]
t2 = dates_sorted.iloc[int(len(dates_sorted) * 0.85)]

train_mask = df["Date"] < t1
val_mask   = (df["Date"] >= t1) & (df["Date"] < t2)
test_mask  = df["Date"] >= t2

# Definicja zmiennej celu
y_train = df.loc[train_mask, "RainTomorrow"].values
y_val   = df.loc[val_mask, "RainTomorrow"].values
y_test  = df.loc[test_mask, "RainTomorrow"].values

# Definicja kolumn
cat_cols = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"]
base_cols = [c for c in df.columns if c not in cat_cols + ["Date", "RainTomorrow", "RainToday", "RainfallTomorrow_mm"]]
all_cols = cat_cols + base_cols

# ZESTAWY DANYCH (KOPIE BAZOWE)
X_train_base = df.loc[train_mask, all_cols].copy()
X_val_base   = df.loc[val_mask, all_cols].copy()
X_test_base  = df.loc[test_mask, all_cols].copy()


# 1. ZESTAW A: DLA CATBOOST (Kategorie jako tekst/category)
X_train_cat = X_train_base.copy()
X_val_cat   = X_val_base.copy()
X_test_cat  = X_test_base.copy()

for col in cat_cols:
    # Wypełnianie braków 'Missing' i ustawianie typu 'category'
    X_train_cat[col] = X_train_cat[col].fillna("Missing").astype("category")
    X_val_cat[col]   = X_val_cat[col].fillna("Missing").astype("category")
    X_test_cat[col]  = X_test_cat[col].fillna("Missing").astype("category")


# 2. ZESTAW B: DLA XGBOOST/LGBM (Kategorie jako liczby - OrdinalEncoder)
X_train_num = X_train_base.copy()
X_val_num   = X_val_base.copy()
X_test_num  = X_test_base.copy()

# Definicja OrdinalEncoder (handle_unknown='use_encoded_value' i unknown_value=-1)
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Fitujemy TYLKO na train i transformujemy (wypełniając wcześniej braki "Missing")
# Używamy .astype(str) dla stabilności kodowania
X_train_num[cat_cols] = encoder.fit_transform(X_train_num[cat_cols].fillna("Missing").astype(str))

# Transformujemy resztę (używając mapowania z train)
X_val_num[cat_cols]   = encoder.transform(X_val_num[cat_cols].fillna("Missing").astype(str))
X_test_num[cat_cols]  = encoder.transform(X_test_num[cat_cols].fillna("Missing").astype(str))

print(f"Rozmiary: Train: {len(X_train_cat)}, Val: {len(X_val_cat)}, Test: {len(X_test_cat)}")


# 4. OPTUNA: KLASYFIKACJA

# XGBoost

def objective_xgb(trial):
    # 1. Parametry do szukania
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }
    
    
    params.update({
        "n_estimators": 1000,
        "tree_method": "hist",
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
        # "early_stopping_rounds": 50  <--- MOŻNA DODAĆ TU w nowym XGBoost, zamiast w .fit()
        # Ale użyjmy callbacku, to bardziej uniwersalne:
        "callbacks": [EarlyStopping(rounds=50, save_best=True)] 
    })
    
    model = XGBClassifier(**params)
    
    # 3. .fit() BEZ early_stopping_rounds jako argumentu
    model.fit(X_train_num, y_train, 
              eval_set=[(X_val_num, y_val)], 
              verbose=False)
              
    return f1_score(y_val, model.predict(X_val_num))

# 4. Optuna z Seedem
sampler = TPESampler(seed=42)
study_xgb = optuna.create_study(direction="maximize", sampler=sampler)
study_xgb.optimize(objective_xgb, n_trials=10)

print(f"XGBoost Best F1: {study_xgb.best_value:.4f}")

# 5. Finalny model
final_xgb_params = study_xgb.best_params.copy()
# Dodajemy stałe parametry I CALLBACK
final_xgb_params.update({
    "n_estimators": 1000,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "callbacks": [EarlyStopping(rounds=50, save_best=True)]
})

best_xgb = XGBClassifier(**final_xgb_params)
# Fit bez early_stopping_rounds (jest w callbacks wewnątrz modelu)
best_xgb.fit(X_train_num, y_train, 
             eval_set=[(X_val_num, y_val)], 
             verbose=False)


# B. LightGBM



def objective_lgbm(trial):
    params = {
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    params.update({
        "n_estimators": 1000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
        "metric": "binary_logloss"
    })
    
    model = LGBMClassifier(**params)
    model.fit(X_train_num, y_train, 
              eval_set=[(X_val_num, y_val)], 
              callbacks=[early_stopping(50, verbose=False)])
              
    return f1_score(y_val, model.predict(X_val_num))

# Używamy tego samego samplera (lub nowego z seed=42)
study_lgbm = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study_lgbm.optimize(objective_lgbm, n_trials=10)
print(f"LightGBM Best F1: {study_lgbm.best_value:.4f}")

# Finalny model LGBM
final_lgbm_params = study_lgbm.best_params.copy()
final_lgbm_params.update({
    "n_estimators": 1000, "random_state": 42, "n_jobs": -1, "verbose": -1, "metric": "binary_logloss"
})
best_lgbm = LGBMClassifier(**final_lgbm_params)
best_lgbm.fit(X_train_num, y_train, 
              eval_set=[(X_val_num, y_val)], 
              callbacks=[early_stopping(50, verbose=False)])


# C. CatBoost

def objective_cat(trial):
    params = {
        "depth": trial.suggest_int("depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
    }
    params.update({
        "iterations": 100, 
        "random_seed": 42,
        "verbose": 0,
        "od_type": "Iter", "od_wait": 50,
        "allow_writing_files": False,
        "cat_features": cat_cols 
    })
    
    model = CatBoostClassifier(**params)
    model.fit(X_train_cat, y_train, eval_set=(X_val_cat, y_val), use_best_model=True)
    return f1_score(y_val, model.predict(X_val_cat))

study_cat = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study_cat.optimize(objective_cat, n_trials=10)
print(f"CatBoost Best F1: {study_cat.best_value:.4f}")

# Finalny model CatBoost
final_cat_params = study_cat.best_params.copy()
final_cat_params.update({
    "iterations": 100, "random_seed": 42, "verbose": 0,
    "od_type": "Iter", "od_wait": 50, "allow_writing_files": False,
    "cat_features": cat_cols 
})
best_cat = CatBoostClassifier(**final_cat_params)
best_cat.fit(X_train_cat, y_train, eval_set=(X_val_cat, y_val), use_best_model=True, verbose=0)



# 5. ENSEMBLE KLASYFIKACJI


# Obliczamy prawdopodobieństwa dla walidacji
p_xgb_val = best_xgb.predict_proba(X_val_num)[:, 1]
p_lgbm_val = best_lgbm.predict_proba(X_val_num)[:, 1]
p_cat_val = best_cat.predict_proba(X_val_cat)[:, 1]

p_ens_val = (p_xgb_val + p_lgbm_val + p_cat_val) / 3

# Szukamy idealnego progu (Threshold Tuning)
best_threshold = 0.5
best_f1 = 0.0

# Sprawdzamy progi od 0.10 do 0.90
for thr in np.linspace(0.1, 0.9, 81):
    # Symulujemy predykcję na walidacji
    y_val_pred_temp = (p_ens_val >= thr).astype(int)
    score = f1_score(y_val, y_val_pred_temp)
    
    if score > best_f1:
        best_f1 = score
        best_threshold = thr

print(f"Znaleziono najlepszy próg (na Val): {best_threshold:.3f} z F1: {best_f1:.4f}")

# Aplikujemy ten próg na tescie (prawdziwa ocena)
p_xgb_test = best_xgb.predict_proba(X_test_num)[:, 1]
p_lgbm_test = best_lgbm.predict_proba(X_test_num)[:, 1]
p_cat_test = best_cat.predict_proba(X_test_cat)[:, 1]

p_ens_test = (p_xgb_test + p_lgbm_test + p_cat_test) / 3

# Używamy 'best_threshold' zamiast 0.5
y_pred_ens_test = (p_ens_test >= best_threshold).astype(int)

print(f"\n--- WYNIKI KOŃCOWE (Test Set) ---")
print(f"Ensemble F1 Score: {f1_score(y_test, y_pred_ens_test):.4f}")
print(f"Ensemble ROC AUC:  {roc_auc_score(y_test, p_ens_test):.4f}")
print(classification_report(y_test, y_pred_ens_test))


# 6. REGRESJA


df = df.sort_values(['Location', 'Date']) 
df["RainfallTomorrow_mm"] = df.groupby("Location")["Rainfall"].shift(-1)

# Wracamy do sortowania chronologicznego (kluczowe dla masek train/val/test)
df = df.sort_values('Date').reset_index(drop=True)

# 2. Filtrowanie (Tylko dni, w których ma padać + mamy dane o jutrze)
# Uwaga: Musimy przeliczyć maski, bo reset_index mógł zmienić indeksy, 
# ale jeśli robisz to w jednym ciągu, to maski z sekcji 3 powinny pasować do dat.
# Dla pewności odświeżam maski na podstawie dat:
train_mask = df["Date"] < t1
val_mask   = (df["Date"] >= t1) & (df["Date"] < t2)
test_mask  = df["Date"] >= t2

reg_mask = (df["RainTomorrow"] == 1) & df["RainfallTomorrow_mm"].notna()

reg_train_mask = train_mask & reg_mask
reg_val_mask   = val_mask   & reg_mask
reg_test_mask  = test_mask  & reg_mask

y_tr_reg = df.loc[reg_train_mask, "RainfallTomorrow_mm"].values
y_val_reg = df.loc[reg_val_mask, "RainfallTomorrow_mm"].values
y_te_reg = df.loc[reg_test_mask, "RainfallTomorrow_mm"].values

# Logarytmowanie celu
y_tr_reg_log = np.log1p(y_tr_reg)
y_val_reg_log = np.log1p(y_val_reg)

# 3. Przygotowanie danych, ponowne użycie encodera
X_tr_reg_num = df.loc[reg_train_mask, cat_cols + base_cols].copy()
X_val_reg_num = df.loc[reg_val_mask, cat_cols + base_cols].copy()
X_te_reg_num = df.loc[reg_test_mask, cat_cols + base_cols].copy()


X_tr_reg_num[cat_cols]  = encoder.transform(X_tr_reg_num[cat_cols].astype(str))
X_val_reg_num[cat_cols] = encoder.transform(X_val_reg_num[cat_cols].astype(str))
X_te_reg_num[cat_cols]  = encoder.transform(X_te_reg_num[cat_cols].astype(str))

# 4. Tuning XGBRegressor
def objective_xgb_reg(trial):
    # Parametry do szukania
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    
    # Parametry stałe + Callback
    params.update({
        "n_estimators": 1000,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "callbacks": [EarlyStopping(rounds=50, save_best=True)]
    })
    
    model = XGBRegressor(**params)
    model.fit(X_tr_reg_num, y_tr_reg_log, 
              eval_set=[(X_val_reg_num, y_val_reg_log)], 
              verbose=False)
    
    # Ewaluacja RMSE (na oryginalnej skali mm)
    preds_log = model.predict(X_val_reg_num)
    preds_mm = np.expm1(preds_log)
    return mean_squared_error(y_val_reg, preds_mm) ** 0.5

sampler_reg = TPESampler(seed=42)
study_xgb_reg = optuna.create_study(direction="minimize", sampler=sampler_reg)
study_xgb_reg.optimize(objective_xgb_reg, n_trials=10)

print(f"Best RMSE (Val): {study_xgb_reg.best_value:.4f}")

# 5. Finalny Model Regresji
final_reg_params = study_xgb_reg.best_params.copy()
final_reg_params.update({
    "n_estimators": 1000,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "callbacks": [EarlyStopping(rounds=50, save_best=True)]
})

best_xgb_reg = XGBRegressor(**final_reg_params)
best_xgb_reg.fit(X_tr_reg_num, y_tr_reg_log, 
                 eval_set=[(X_val_reg_num, y_val_reg_log)], 
                 verbose=False)

# 6. Test Końcowy
preds_log_test = best_xgb_reg.predict(X_te_reg_num)
preds_mm_test = np.expm1(preds_log_test)

rmse_test = mean_squared_error(y_te_reg, preds_mm_test) ** 0.5
mae_test = mean_absolute_error(y_te_reg, preds_mm_test)

print("\n=== WYNIKI REGRESJI (Test Set) ===")
print(f"RMSE: {rmse_test:.3f} mm")
print(f"MAE:  {mae_test:.3f} mm")
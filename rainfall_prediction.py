import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import optuna
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error



df = pd.read_csv('weatherAUS.csv')
df.head()
# convert date to datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# sort data by date and location
df = df.sort_values(['Date', 'Location'], ascending = [True, True]).reset_index(drop = True)

# drop missing target values
df = df.dropna(subset = ['RainTomorrow']).copy()

# to binary
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})

# clamp
def clamp(series, lo=None, hi=None):
    if lo is not None:
        series = series.clip(lower = lo)
    if hi is not None:
        series = series.clip(upper = hi)
    return series

for col in ['Humidity9am', 'Humidity3pm']:
    df[col] = clamp(df[col], 0, 100)
    
for col in ['Cloud9am', 'Cloud3pm']:
    df[col] = clamp(df[col], 0, 9)
    
nonneg_cols = [
"Rainfall", "Evaporation", "Pressure9am", "Pressure3pm",
"WindGustSpeed", "WindSpeed9am", "WindSpeed3pm"
]

for col in nonneg_cols:
    df[col] = clamp(df[col], 0, None)
    
df['RainToday_bin'] = df['RainToday'].map({'Yes': 1, 'No': 0})


# Temperature difference between max and min (daily range)
df["temp_range"] = df["MaxTemp"] - df["MinTemp"]

# Change in temperature between 9am and 3pm
df["temp_delta_9to3"] = df["Temp3pm"] - df["Temp9am"]

# Change in humidity between morning and afternoon
df["humidity_delta_9to3"] = df["Humidity3pm"] - df["Humidity9am"]

# Pressure change from morning to afternoon
df["pressure_delta_9to3"] = df["Pressure3pm"] - df["Pressure9am"]

# Wind speed change between morning and afternoon
df["wind_delta_9to3"] = df["WindSpeed3pm"] - df["WindSpeed9am"]

dayofyear = df["Date"].dt.dayofyear
df["doy_sin"] = np.sin(2 * np.pi * dayofyear / 365.0)
df["doy_cos"] = np.cos(2 * np.pi * dayofyear / 365.0)

df["Month"] = df["Date"].dt.month


dates_sorted = df['Date'].sort_values()
t1 = dates_sorted[int(len(dates_sorted) * 0.7)]
t2 = dates_sorted[int(len(dates_sorted) * 0.85)]

train_mask = df["Date"] < t1
val_mask   = (df["Date"] >= t1) & (df["Date"] < t2)
test_mask  = df["Date"] >= t2

y_train = df.loc[train_mask, "RainTomorrow"].values
y_val   = df.loc[val_mask, "RainTomorrow"].values
y_test  = df.loc[test_mask, "RainTomorrow"].values



cat_cols = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm"]


num_cols = [
    "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
    "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm",
    "Pressure9am", "Pressure3pm",
    "Cloud9am", "Cloud3pm",
    "Temp9am", "Temp3pm",
    "RainToday_bin",
    "temp_range", "temp_delta_9to3", "humidity_delta_9to3",
    "pressure_delta_9to3", "wind_delta_9to3",
    "doy_sin", "doy_cos", "Month"
]

X_train = df.loc[train_mask, cat_cols + num_cols]
X_val = df.loc[val_mask, cat_cols + num_cols]
X_test = df.loc[test_mask, cat_cols + num_cols]

for col in cat_cols:
    X_train[col] = X_train[col].astype("category")
    X_val[col]   = X_val[col].astype("category")
    X_test[col]  = X_test[col].astype("category")
    X_train[col] = X_train[col].cat.codes
    X_val[col]   = X_val[col].cat.codes
    X_test[col]  = X_test[col].cat.codes

# alternatively, data imputation


# pipeline for numeric columns
#numeric_transformer = Pipeline(steps=[
#    ("imputer", KNNImputer(n_neighbors=3))
#])

#categorical_transformer = Pipeline(steps=[
#    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
#])

#preprocessor = ColumnTransformer(
#    transformers=[
#        ("num", numeric_transformer, num_cols),
#        ("cat", categorical_transformer, cat_cols)
#    ],
#    remainder="drop",
#    verbose_feature_names_out=False
#)


#X_train_copy = X_train.copy()
#X_val_copy   = X_val.copy()
#X_test_copy  = X_test.copy()


# fit only on training data
#preprocessor.fit(X_train_copy)

# transform all splits
#X_train_prep = preprocessor.transform(X_train_copy)
#X_val_prep   = preprocessor.transform(X_val_copy)
#X_test_prep  = preprocessor.transform(X_test_copy)



# models, without data imputation

Xtr = X_train.copy()
ytr = y_train.copy()
Xval = X_val.copy()
yval = y_val.copy()

def objective_xgb(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "random_state": 42,
        "use_label_encoder": False,
        "eval_metric": "logloss",
    }
    model = XGBClassifier(**params)
    score = cross_val_score(model, Xtr, ytr, scoring="f1", cv=3, n_jobs=-1).mean()
    return score

def objective_lgbm(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "random_state": 42,
    }
    model = LGBMClassifier(**params)
    score = cross_val_score(model, Xtr, ytr, scoring="f1", cv=3, n_jobs=-1).mean()
    return score

def objective_cat(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 600),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "random_seed": 42,
        "verbose": 0
    }
    model = CatBoostClassifier(**params)
    score = cross_val_score(model, Xtr, ytr, scoring="f1", cv=3, n_jobs=-1).mean()
    return score


# Running optuna

results = []

print(f" OPTUNA TUNING: Boosting models on raw data")

# XGBoost
study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_xgb, n_trials=20)
best_xgb = XGBClassifier(**study_xgb.best_params, use_label_encoder=False, eval_metric="logloss")
best_xgb.fit(Xtr, ytr)
preds_xgb = best_xgb.predict(Xval)
proba_xgb = best_xgb.predict_proba(Xval)[:,1]
results.append({
    "Model": "XGBoost",
    "F1": f1_score(yval, preds_xgb),
    "AUC": roc_auc_score(yval, proba_xgb),
    "Acc": accuracy_score(yval, preds_xgb),
    "BestParams": study_xgb.best_params
})

# LightGBM
study_lgbm = optuna.create_study(direction="maximize")
study_lgbm.optimize(objective_lgbm, n_trials=20)
best_lgbm = LGBMClassifier(**study_lgbm.best_params)
best_lgbm.fit(Xtr, ytr)
preds_lgbm = best_lgbm.predict(Xval)
proba_lgbm = best_lgbm.predict_proba(Xval)[:,1]
results.append({
    "Model": "LightGBM",
    "F1": f1_score(yval, preds_lgbm),
    "AUC": roc_auc_score(yval, proba_lgbm),
    "Acc": accuracy_score(yval, preds_lgbm),
    "BestParams": study_lgbm.best_params
})

# CatBoost
study_cat = optuna.create_study(direction="maximize")
study_cat.optimize(objective_cat, n_trials=20)
best_cat = CatBoostClassifier(**study_cat.best_params)
best_cat.fit(Xtr, ytr)
preds_cat = best_cat.predict(Xval)
proba_cat = best_cat.predict_proba(Xval)[:,1]
results.append({
    "Model": "CatBoost",
    "F1": f1_score(yval, preds_cat),
    "AUC": roc_auc_score(yval, proba_cat),
    "Acc": accuracy_score(yval, preds_cat),
    "BestParams": study_cat.best_params
})


# Final raport

df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="F1", ascending=False).reset_index(drop=True)

print("FINAL RESULTS")
print(df_results)




# Get predictions (probabilities of class "1")
proba_xgb = best_xgb.predict_proba(Xval)[:, 1]
proba_lgbm = best_lgbm.predict_proba(Xval)[:, 1]
proba_cat = best_cat.predict_proba(Xval)[:, 1]

# Weighted average (you can also do simple averaging)
ensemble_proba = (proba_xgb + proba_lgbm + proba_cat) / 3

# Final decision using threshold 0.5
ensemble_pred = (ensemble_proba >= 0.5).astype(int)

# Evaluation
f1 = f1_score(yval, ensemble_pred)
auc = roc_auc_score(yval, ensemble_proba)
acc = accuracy_score(yval, ensemble_pred)

print(" Ensemble Results (XGB + LGBM + CAT)")
print(f"F1:  {f1:.4f}")
print(f"AUC: {auc:.4f}")
print(f"ACC: {acc:.4f}")

print("\nClassification report:")
print(classification_report(yval, ensemble_pred, digits=3))

print("\nConfusion matrix:")
print(confusion_matrix(yval, ensemble_pred))








# regression target: how many mm of rainfall will fall tomorrow
# (Rainfall shifted up by 1 day within each location)
df["RainfallTomorrow_mm"] = df.groupby("Location")["Rainfall"].shift(-1)

# regression masks: only observations where:
# - they are in train/val/test
# - tomorrow it rains (RainTomorrow == 1)
# - we know tomorrow's rainfall amount (RainfallTomorrow_mm is not NaN)
reg_train_mask = train_mask & (df["RainTomorrow"] == 1) & df["RainfallTomorrow_mm"].notna()
reg_val_mask   = val_mask   & (df["RainTomorrow"] == 1) & df["RainfallTomorrow_mm"].notna()
reg_test_mask  = test_mask  & (df["RainTomorrow"] == 1) & df["RainfallTomorrow_mm"].notna()

# regression features – the same as for classification (cat_cols + num_cols)
X_train_reg = df.loc[reg_train_mask, cat_cols + num_cols].copy()
X_val_reg   = df.loc[reg_val_mask,   cat_cols + num_cols].copy()
X_test_reg  = df.loc[reg_test_mask,  cat_cols + num_cols].copy()

y_train_reg = df.loc[reg_train_mask, "RainfallTomorrow_mm"].values
y_val_reg   = df.loc[reg_val_mask,   "RainfallTomorrow_mm"].values
y_test_reg  = df.loc[reg_test_mask,  "RainfallTomorrow_mm"].values

# encode categorical variables same as before (cat.codes)
for col in cat_cols:
    X_train_reg[col] = X_train_reg[col].astype("category").cat.codes
    X_val_reg[col]   = X_val_reg[col].astype("category").cat.codes
    X_test_reg[col]  = X_test_reg[col].astype("category").cat.codes

# working copies for Optuna
Xtr_reg = X_train_reg.copy()
ytr_reg = y_train_reg.copy()
Xval_reg = X_val_reg.copy()
yval_reg = y_val_reg.copy()



# Optuna objective functions for regression (RMSE)
# using scoring="neg_root_mean_squared_error", so direction="maximize"

def objective_xgb_reg(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 3.0),
        "random_state": 42,
    }
    model = XGBRegressor(**params)
    score = cross_val_score(
        model,
        Xtr_reg,
        ytr_reg,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1
    ).mean()
    return score  # the closer to 0 (less negative), the better


def objective_lgbm_reg(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600),
        "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "random_state": 42,
    }
    model = LGBMRegressor(**params)
    score = cross_val_score(
        model,
        Xtr_reg,
        ytr_reg,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1
    ).mean()
    return score


def objective_cat_reg(trial):
    params = {
        "iterations": trial.suggest_int("iterations", 200, 600),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True),
        "random_seed": 42,
        "verbose": 0,
        "loss_function": "RMSE",
    }
    model = CatBoostRegressor(**params)
    score = cross_val_score(
        model,
        Xtr_reg,
        ytr_reg,
        scoring="neg_root_mean_squared_error",
        cv=3,
        n_jobs=-1
    ).mean()
    return score



# optuna tuning – regression models (XGB / LGBM / CatBoost)

reg_results = []

print("\n OPTUNA TUNING: regression models (rainfall amount for rainy days)")

# XGBRegressor
study_xgb_reg = optuna.create_study(direction="maximize")
study_xgb_reg.optimize(objective_xgb_reg, n_trials=20)
best_xgb_reg = XGBRegressor(**study_xgb_reg.best_params)
best_xgb_reg.fit(Xtr_reg, ytr_reg)

preds_xgb_reg_val = best_xgb_reg.predict(Xval_reg)
mae_xgb = mean_absolute_error(yval_reg, preds_xgb_reg_val)
rmse_xgb = mean_squared_error(yval_reg, preds_xgb_reg_val) ** 0.5

reg_results.append({
    "Model": "XGBRegressor",
    "MAE_val": mae_xgb,
    "RMSE_val": rmse_xgb,
    "BestParams": study_xgb_reg.best_params
})


# LGBMRegressor
study_lgbm_reg = optuna.create_study(direction="maximize")
study_lgbm_reg.optimize(objective_lgbm_reg, n_trials=20)
best_lgbm_reg = LGBMRegressor(**study_lgbm_reg.best_params)
best_lgbm_reg.fit(Xtr_reg, ytr_reg)

preds_lgbm_reg_val = best_lgbm_reg.predict(Xval_reg)
mae_lgbm = mean_absolute_error(yval_reg, preds_lgbm_reg_val)
rmse_lgbm = mean_squared_error(yval_reg, preds_lgbm_reg_val) ** 0.5

reg_results.append({
    "Model": "LGBMRegressor",
    "MAE_val": mae_lgbm,
    "RMSE_val": rmse_lgbm,
    "BestParams": study_lgbm_reg.best_params
})


# CatBoostRegressor
study_cat_reg = optuna.create_study(direction="maximize")
study_cat_reg.optimize(objective_cat_reg, n_trials=20)
best_cat_reg = CatBoostRegressor(**study_cat_reg.best_params)
best_cat_reg.fit(Xtr_reg, ytr_reg)

preds_cat_reg_val = best_cat_reg.predict(Xval_reg)
mae_cat = mean_absolute_error(yval_reg, preds_cat_reg_val)
rmse_cat = mean_squared_error(yval_reg, preds_cat_reg_val) ** 0.5

reg_results.append({
    "Model": "CatBoostRegressor",
    "MAE_val": mae_cat,
    "RMSE_val": rmse_cat,
    "BestParams": study_cat_reg.best_params
})


# final report – regression models

df_reg_results = pd.DataFrame(reg_results)
df_reg_results = df_reg_results.sort_values(by="RMSE_val", ascending=True).reset_index(drop=True)

print("\nFINAL REGRESSION RESULTS (validation, rainy days only)")
print(df_reg_results)



# regression ensemble (average of XGB/LGBM/CatBoost)

ensemble_reg_val = (
    preds_xgb_reg_val +
    preds_lgbm_reg_val +
    preds_cat_reg_val
) / 3

mae_ens = mean_absolute_error(yval_reg, ensemble_reg_val)
rmse_ens = mean_squared_error(yval_reg, ensemble_reg_val) ** 0.5

print("\n Ensemble regression (XGBReg + LGBMReg + CatBoostReg) on validation")
print(f"MAE:  {mae_ens:.3f} mm")
print(f"RMSE: {rmse_ens:.3f} mm")



# optional: evaluation on test set for best models and ensemble

preds_xgb_reg_test  = best_xgb_reg.predict(X_test_reg)
preds_lgbm_reg_test = best_lgbm_reg.predict(X_test_reg)
preds_cat_reg_test  = best_cat_reg.predict(X_test_reg)

ensemble_reg_test = (
    preds_xgb_reg_test +
    preds_lgbm_reg_test +
    preds_cat_reg_test
) / 3

mae_test = mean_absolute_error(y_test_reg, ensemble_reg_test)
rmse_test = mean_squared_error(y_test_reg, ensemble_reg_test) ** 0.5

print("\n Ensemble regression on TEST (rainy days only)")
print(f"MAE test:  {mae_test:.3f} mm")
print(f"RMSE test: {rmse_test:.3f} mm")
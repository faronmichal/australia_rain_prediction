import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

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


# --- Temperature difference between max and min (daily range)
df["temp_range"] = df["MaxTemp"] - df["MinTemp"]

# --- Change in temperature between 9am and 3pm
df["temp_delta_9to3"] = df["Temp3pm"] - df["Temp9am"]

# --- Change in humidity between morning and afternoon
df["humidity_delta_9to3"] = df["Humidity3pm"] - df["Humidity9am"]

# --- Pressure change from morning to afternoon
df["pressure_delta_9to3"] = df["Pressure3pm"] - df["Pressure9am"]

# --- Wind speed change between morning and afternoon
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


# alternatively, data imputation


# pipeline for numeric columns
numeric_transformer = Pipeline(steps=[
    ("imputer", KNNImputer(n_neighbors=5))
])

categorical_transformer = Pipeline(steps=[
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

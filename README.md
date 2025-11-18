Two-Stage Rainfall Prediction Project

This project implements a robust two-stage machine learning pipeline to forecast rainfall in Australia using the weatherAUS.csv dataset.

Key Features

    Two-Stage Pipeline: Separates the problem into Classification (Rain/No Rain) and Regression (Rainfall Amount).

    Ensemble Modeling: Uses optimized XGBoost, LightGBM, and CatBoost classifiers.

    Optuna Tuning: Hyperparameters are tuned using Optuna (F1-Score for classification, RMSE for regression).

    Time-Series Validation: Uses a strict chronological split for realistic testing.

    Advanced Features: Includes Lag features (past weather data) and cyclical date encoding.

Deployment Strategy

The model is prepared for production by wrapping all preprocessing steps (encoding, feature engineering) and all four trained models (3 Classifiers, 1 Regressor) into a single Production Wrapper Class. This single object is saved using joblib for easy deployment and consistent predictions.

Dependencies

Requires numpy, pandas, scikit-learn, optuna, xgboost, lightgbm, and catboost.

"""Shared feature definitions so training and the API prepare data identically."""
import numpy as np
import pandas as pd

# These are the eight predictors provided by the diabetes dataset.
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
LOG_FEATURES = ["DiabetesPedigreeFunction", "Age"]          # right-skewed -> log scale
ZERO_IS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select model features and apply the log transforms."""
    X = df[FEATURES].astype(float).copy()
    for c in LOG_FEATURES:
        X[c] = np.log(X[c])
    return X

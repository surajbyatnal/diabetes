"""
Diabetes risk API (Flask).

    python app.py            ->  http://127.0.0.1:5000

Endpoints
    GET  /api/health        service status
    GET  /api/model-info    metrics, risk bands, ranges (for the website)
    POST /api/predict       eight diabetes-dataset predictors -> risk estimate
The same server also serves the frontend at /, so one command is enough.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from features import FEATURES, build_matrix
from train_model import INFO_PATH, MODEL_PATH, train

HERE = Path(__file__).resolve().parent
FRONTEND = HERE.parent / "frontend"

# Train on first run if no saved model exists (or if it can't be loaded, e.g. new scikit-learn).
def load_model():
    if MODEL_PATH.exists() and INFO_PATH.exists():
        try:
            return joblib.load(MODEL_PATH), json.loads(INFO_PATH.read_text())
        except Exception as exc:                     # noqa: BLE001
            print(f"Could not load saved model ({exc}); retraining...")
    print("Training model...")
    info = train()
    return joblib.load(MODEL_PATH), info

MODEL, INFO = load_model()
SCALER, CLF = MODEL.named_steps["scale"], MODEL.named_steps["clf"]
THRESH = INFO["thresholds"]

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/api/*": {"origins": "*"}})   # lets a separately-served frontend call the API

LIMITS = {
    "pregnancies": (0, 17, True),
    "glucose": (40, 400, False),
    "blood_pressure": (20, 140, True),
    "skin_thickness": (1, 100, True),
    "insulin": (1, 900, True),
    "bmi": (15, 70, False),
    "diabetes_pedigree_function": (0.01, 3, False),
    "age": (21, 81, True),
}


def validate(payload):
    """Return (clean_values, errors). Glucose is expected in mg/dL."""
    errors, clean = {}, {}
    if not isinstance(payload, dict):
        return {}, {"_": "Send a JSON object."}
    for key, (lo, hi, integer) in LIMITS.items():
        raw = payload.get(key)
        try:
            val = float(raw)
            if not np.isfinite(val):
                raise ValueError
        except (TypeError, ValueError):
            errors[key] = f"{key} is required and must be a number."
            continue
        if integer and val != int(val):
            errors[key] = f"{key} must be a whole number."
        elif not lo <= val <= hi:
            errors[key] = f"{key} must be between {lo} and {hi}."
        else:
            clean[key] = int(val) if integer else val
    return clean, errors


def band_of(p):
    return "low" if p < THRESH["low"] else "moderate" if p < THRESH["high"] else "high"


def score(c):
    input_features = {
        "pregnancies": "Pregnancies",
        "glucose": "Glucose",
        "blood_pressure": "BloodPressure",
        "skin_thickness": "SkinThickness",
        "insulin": "Insulin",
        "bmi": "BMI",
        "diabetes_pedigree_function": "DiabetesPedigreeFunction",
        "age": "Age",
    }
    used = {
        feature: min(INFO["ranges"][feature][1],
                     max(INFO["ranges"][feature][0], c[key]))
        for key, feature in input_features.items()
    }
    row = pd.DataFrame([used])
    X = build_matrix(row)
    p = float(MODEL.predict_proba(X)[0, 1])
    # per-feature push on the log-odds, relative to the average person in the study
    z = SCALER.transform(MODEL.named_steps["impute"].transform(X))[0]
    contrib = z * CLF.coef_[0]
    shown = {feature: c[key] for key, feature in input_features.items()}
    parts = [{"feature": f, "value": shown[f], "contribution": float(contrib[i]),
              "average": INFO["avg_raw"].get(f)} for i, f in enumerate(FEATURES)]
    clamped = any(used[feature] != c[key] for key, feature in input_features.items())
    return p, parts, clamped


@app.get("/api/health")
def health():
    return jsonify(status="ok", model="logistic_regression", features=FEATURES)


@app.get("/api/model-info")
def model_info():
    return jsonify(INFO)


@app.post("/api/predict")
def predict():
    clean, errors = validate(request.get_json(silent=True))
    if errors:
        return jsonify(error="Please check your answers.", fields=errors), 400
    p, parts, clamped = score(clean)
    return jsonify(
        probability=p, percent=round(p * 100), band=band_of(p),
        thresholds=THRESH, contributions=parts,
        alerts=dict(glucose_in_diabetes_range=clean["glucose"] >= 200, values_capped=clamped),
    )


# ---- serve the frontend from the same server (optional convenience) ----
@app.get("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

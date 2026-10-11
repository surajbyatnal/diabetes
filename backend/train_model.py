"""
Train and evaluate the diabetes risk model.

    python train_model.py                      # uses data/diabetes.csv
    python train_model.py path/to/diabetes.csv

Writes:
    model/model.joblib      the fitted scikit-learn pipeline (used by the API)
    model/model_info.json   metrics, risk bands, ranges (shown on the website)
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import FEATURES, LOG_FEATURES, ZERO_IS_MISSING, build_matrix

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
INFO_PATH = MODEL_DIR / "model_info.json"

LOW, HIGH = 0.25, 0.50          # risk-band cut-offs on the predicted probability


def make_model() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=2000)),
    ])


def train(data_path: Path = HERE / "data" / "diabetes.csv") -> dict:
    df = pd.read_csv(data_path)

    # 0 is not a real value for these columns: it means "not recorded"
    for c in ZERO_IS_MISSING:
        df[c] = df[c].replace(0, np.nan)

    X, y = build_matrix(df), df["Outcome"]

    # 1) Hold-out test: 20% of rows never seen during training
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    m = make_model().fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    yh = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, yh).ravel()
    holdout = dict(n_test=int(len(yte)), accuracy=accuracy_score(yte, yh), auc=roc_auc_score(yte, p),
                   sensitivity=recall_score(yte, yh), precision=precision_score(yte, yh),
                   f1=f1_score(yte, yh), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))

    # 2) Repeated 5-fold cross-validation on all rows: a steadier estimate than one small split
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=1)
    r = cross_validate(make_model(), X, y, cv=cv,
                       scoring=["accuracy", "roc_auc", "recall", "precision"])
    cvres = {k: float(r[f"test_{k}"].mean()) for k in ["accuracy", "roc_auc", "recall", "precision"]}
    cvres["accuracy_sd"] = float(r["test_accuracy"].std())
    cvres["auc_sd"] = float(r["test_roc_auc"].std())

    # 3) Do the risk bands mean what they say? (out-of-fold predictions)
    oof = cross_val_predict(make_model(), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=7),
                            method="predict_proba")[:, 1]
    bands = []
    for name, lo, hi in [("low", 0, LOW), ("moderate", LOW, HIGH), ("high", HIGH, 1.01)]:
        mask = (oof >= lo) & (oof < hi)
        bands.append(dict(band=name, lo=lo, hi=min(hi, 1.0), share=float(mask.mean()),
                          observed_rate=float(y[mask].mean()), n=int(mask.sum())))
    flag = oof >= LOW
    screen = dict(sensitivity=float(flag[y == 1].mean()), specificity=float((~flag)[y == 0].mean()),
                  missed_share=float((~flag)[y == 1].mean()))
    fpr, tpr, _ = roc_curve(y, oof)
    idx = np.linspace(0, len(fpr) - 1, 40).astype(int)

    # 4) Final model: refit on ALL rows
    final = make_model().fit(X, y)
    clf = final.named_steps["clf"]
    scaler = final.named_steps["scale"]

    info = dict(
        features=FEATURES,
        thresholds=dict(low=LOW, high=HIGH),
        bands=bands,
        ranges={c: [float(df[c].min()), float(df[c].max())] for c in FEATURES},
        avg_raw={c: float(df[c].mean()) for c in FEATURES},
        coef=[float(v) for v in clf.coef_[0]],
        # lets the website show "how you compare with the dataset" (percentiles 0..100)
        distributions={c: [float(v) for v in np.quantile(df[c].dropna(), np.linspace(0, 1, 101))]
                       for c in FEATURES},
        # plain numbers of the fitted model (used only by the browser-only demo build)
        model_params=dict(mean=[float(v) for v in scaler.mean_], scale=[float(v) for v in scaler.scale_],
                          coef=[float(v) for v in clf.coef_[0]], intercept=float(clf.intercept_[0]),
                          log_features=LOG_FEATURES),
        metrics=dict(holdout=holdout, cv=cvres, screen=screen),
        roc=dict(fpr=[float(fpr[i]) for i in idx], tpr=[float(tpr[i]) for i in idx]),
        data=dict(n=int(len(df)), positives=int(y.sum()), prevalence=float(y.mean())),
    )

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(final, MODEL_PATH)
    INFO_PATH.write_text(json.dumps(info, indent=1))
    return info


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "data" / "diabetes.csv"
    info = train(path)
    ho, cv, sc = info["metrics"]["holdout"], info["metrics"]["cv"], info["metrics"]["screen"]
    print(f"Rows: {info['data']['n']}  (diabetes: {info['data']['prevalence']:.0%})")
    print(f"Hold-out test ({ho['n_test']} rows): accuracy {ho['accuracy']:.3f}  AUC {ho['auc']:.3f}  "
          f"sensitivity {ho['sensitivity']:.3f}  precision {ho['precision']:.3f}")
    print(f"5x10 cross-validation:  accuracy {cv['accuracy']:.3f}  AUC {cv['roc_auc']:.3f}")
    for b in info["bands"]:
        print(f"  band {b['band']:8s} {b['share']:.0%} of people -> {b['observed_rate']:.0%} had diabetes")
    print(f"'Moderate or higher' catches {sc['sensitivity']:.0%} of real cases")
    print(f"Saved {MODEL_PATH.relative_to(HERE)} and {INFO_PATH.relative_to(HERE)}")

"""
train_rainfall_threshold.py
Fits two things from the real rainfall training data:

1. The classic Intensity-Duration power-law threshold curve
   (I = a * D^-b), the standard rainfall-threshold formulation used in
   landslide early-warning literature (Caine 1980 and successors) - fit
   via log-log linear regression on the real triggering events only.

2. A logistic regression classifier on the fuller feature set (cumulative
   rainfall at multiple windows + duration + intensity) for a live
   trigger probability, not just a binary above/below-curve check.

Usage:
    python train_rainfall_threshold.py \
        --data rainfall_training_data.csv \
        --out rainfall_threshold_model.pkl
"""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

FEATURES = ["cum_1d", "cum_3d", "cum_7d", "cum_15d", "duration_days", "storm_intensity_mm_day"]


def fit_id_curve(df):
    """Classic power-law threshold: log(I) = log(a) - b*log(D), fit only
    on real triggering events (label == 1). This is the literal
    'Rainfall I-D Threshold' the problem statement asks for."""
    triggers = df[df["label"] == 1]
    log_d = np.log(triggers["duration_days"])
    log_i = np.log(triggers["storm_intensity_mm_day"].clip(lower=0.1))  # avoid log(0)

    # Simple least-squares fit: log_i = log_a - b*log_d
    A = np.vstack([log_d, np.ones(len(log_d))]).T
    slope, intercept = np.linalg.lstsq(A, log_i, rcond=None)[0]
    b = -slope
    a = np.exp(intercept)
    return {"a": float(a), "b": float(b)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)

    id_curve = fit_id_curve(df)
    print(f"Fitted I-D threshold curve: I = {id_curve['a']:.2f} * D^-{id_curve['b']:.3f}")

    X = df[FEATURES]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    cv_auc = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
    print(f"Logistic regression CV AUC: {cv_auc:.3f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nTest set classification report:")
    print(classification_report(y_test, y_pred))
    print(f"Test AUC: {roc_auc_score(y_test, y_prob):.3f}")

    coefficients = dict(zip(FEATURES, model.coef_[0]))
    print("\nLogistic regression coefficients:")
    for feat, coef in sorted(coefficients.items(), key=lambda x: -abs(x[1])):
        print(f"  {feat}: {coef:.4f}")

    joblib.dump(
        {"model": model, "features": FEATURES, "id_curve": id_curve},
        args.out,
    )
    print(f"\nSaved model to {args.out}")

    # Also save the I-D curve alone as plain JSON - lets Member 3/4/5 use
    # the threshold curve directly (e.g. in the web dashboard or mobile
    # app) without needing to load a pickled sklearn model.
    curve_path = args.out.replace(".pkl", "_id_curve.json")
    with open(curve_path, "w") as f:
        json.dump(id_curve, f, indent=2)
    print(f"Saved I-D curve parameters to {curve_path}")


if __name__ == "__main__":
    main()

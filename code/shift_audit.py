"""Read-only train/test covariate-shift audit for the Satellite Health paper.

The domain label is 0 for competition train rows and 1 for competition test
rows. It is never the satellite-health target. All preprocessing is fitted
inside each domain-classification CV fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "time_since_launch",
    "orbital_altitude",
    "battery_voltage",
    "solar_panel_temperature",
    "attitude_control_error",
    "data_transmission_rate",
    "thermal_control_status",
]
SEED = 20261010


def prepare_shift_data(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return only named telemetry features and the train-vs-test domain label."""
    missing = [name for name in features if name not in train or name not in test]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    train_x = train.loc[:, features].apply(pd.to_numeric, errors="raise").astype(float)
    test_x = test.loc[:, features].apply(pd.to_numeric, errors="raise").astype(float)
    x = pd.concat([train_x, test_x], ignore_index=True)
    domain = np.r_[np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]
    return x, domain


def summarize_missingness(
    train: pd.DataFrame, test: pd.DataFrame, features: list[str]
) -> dict[str, dict[str, float]]:
    """Report test minus train missingness using each split's own denominator."""
    result = {}
    for name in features:
        train_rate = float(train[name].isna().mean())
        test_rate = float(test[name].isna().mean())
        result[name] = {
            "train_rate": train_rate,
            "test_rate": test_rate,
            "difference_test_minus_train": test_rate - train_rate,
        }
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_bootstrap_auc_interval(y: np.ndarray, scores: np.ndarray, seed: int) -> list[float]:
    """Descriptive interval conditional on fixed OOF predictions."""
    rng = np.random.default_rng(seed)
    negatives = np.flatnonzero(y == 0)
    positives = np.flatnonzero(y == 1)
    values = np.empty(1000, dtype=float)
    for index in range(len(values)):
        sample = np.r_[
            rng.choice(negatives, size=len(negatives), replace=True),
            rng.choice(positives, size=len(positives), replace=True),
        ]
        values[index] = roc_auc_score(y[sample], scores[sample])
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def audit(train_path: Path, test_path: Path) -> dict:
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    x, domain = prepare_shift_data(train, test, FEATURES)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    models = {
        "median_scaled_logistic": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED),
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, max_leaf_nodes=15, learning_rate=0.05, random_state=SEED
        ),
    }
    model_results = {}
    for name, model in models.items():
        probabilities = cross_val_predict(
            model, x, domain, cv=cv, method="predict_proba", n_jobs=1
        )[:, 1]
        model_results[name] = {
            "oof_auc": float(roc_auc_score(domain, probabilities)),
            "row_bootstrap_95pct_ci_conditional_on_oof_predictions": _row_bootstrap_auc_interval(
                domain, probabilities, SEED + len(name)
            ),
        }
    return {
        "task": "distinguish competition train rows from test rows using seven telemetry features",
        "interpretation_limit": "Covariate shift only; test health labels and test F1 are unknown. The bootstrap interval conditions on fixed OOF predictions.",
        "python": sys.version.split()[0],
        "python_bits": 64 if sys.maxsize > 2**32 else 32,
        "platform": platform.platform(),
        "sklearn": sklearn.__version__,
        "seed": SEED,
        "cv": "5-fold stratified, shuffled; identical folds for both models",
        "features": FEATURES,
        "rows_train": len(train),
        "rows_test": len(test),
        "train_sha256": _sha256(train_path),
        "test_sha256": _sha256(test_path),
        "missingness": summarize_missingness(train, test, FEATURES),
        "models": model_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {args.output}")
    report = audit(args.data_dir / "train.csv", args.data_dir / "test.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": report["models"]}))


if __name__ == "__main__":
    main()

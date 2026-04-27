"""Train supervised health-risk classification models.

Usage
-----
From the air-monitor/ directory:

    python -m ml.train_classification --db data/air_monitor.duckdb
    python -m ml.train_classification --input data/microbatches.csv
    python -m ml.train_classification --db data/air_monitor.duckdb --system
    python -m ml.train_classification --db data/air_monitor.duckdb --temporal

Scientific design
-----------------
- Temporal train/test split (first 80% / last 20% chronologically) prevents
  future data leaking into training — required for time-series sensor data.
- TimeSeriesSplit cross-validation (N_CV_SPLITS folds) on the training portion
  reports mean +/- std across folds, not a single lucky split.
- ThresholdBaseline: deterministic rule-based classifier included as the
  zero-intelligence baseline in every CV fold and the final test evaluation.
- McNemar's test compares all classifier pairs for statistical significance.
- Health risk thresholds follow WHO 2021 / ASHRAE / EN 15251 standards.

Models: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier,
        ThresholdBaseline (rule-based, no training).

Reports saved:
  outputs/reports/classification_cv_{ts}.json/csv       -- CV mean+/-std
  outputs/reports/classification_final_{ts}.json/csv    -- held-out test metrics
  outputs/reports/classification_mcnemar_{ts}.json/csv  -- pairwise significance
"""

import argparse
import sys
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from ml.config import (
    CORE_SENSOR_FIELDS, DEFAULT_DB_PATH, HEALTH_THRESHOLDS,
    N_CV_SPLITS, OPTIONAL_SYSTEM_FIELDS, ROLLING_WINDOW,
)
from ml.evaluate import classification_report_full, cv_summary, mcnemar_test, save_report
from ml.feature_engineering import (
    add_rolling_features, add_temporal_features,
    build_batch_features, get_batch_timestamps,
    load_from_csv, load_from_duckdb,
)
from ml.labels import make_health_risk_labels
from ml.metrics import measure_inference, measure_training
from ml.model_registry import save_model

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

_CLASSIFIERS = {
    "logistic_regression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    ),
    "decision_tree": DecisionTreeClassifier(
        max_depth=5, class_weight="balanced", random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=50, max_depth=5, class_weight="balanced", random_state=42
    ),
}


class _ThresholdBaseline:
    """Deterministic rule-based classifier — no training, reproduces HEALTH_THRESHOLDS.

    Operates on unscaled feature values. Included in every CV fold and final
    evaluation as the zero-intelligence baseline the ML models must beat.
    """
    def __init__(self, feature_names: list[str]) -> None:
        self.feature_names = list(feature_names)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "_ThresholdBaseline":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(X, columns=self.feature_names)
        risk = pd.Series(0, index=df.index)
        for field, bounds in HEALTH_THRESHOLDS.items():
            if "max" in bounds:
                col = f"{field}_max"
                if col in df.columns:
                    risk |= (df[col] > bounds["max"]).fillna(False).astype(int)
            if "min" in bounds:
                col = f"{field}_min"
                if col in df.columns:
                    risk |= (df[col] < bounds["min"]).fillna(False).astype(int)
        return risk.values


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _impute_nan(X: np.ndarray) -> np.ndarray:
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = col_means[nan_idx[1]]
    return X


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    # --- Load data ----------------------------------------------------------
    if args.input:
        raw = load_from_csv(args.input)
        print(f"Loaded {len(raw)} observations from {args.input}")
    else:
        db = args.db or DEFAULT_DB_PATH
        raw = load_from_duckdb(db, include_system=args.system)
        print(f"Loaded {len(raw)} observations from {db}")

    # --- Feature engineering ------------------------------------------------
    fields = list(CORE_SENSOR_FIELDS)
    if args.system:
        fields += OPTIONAL_SYSTEM_FIELDS

    features = build_batch_features(raw, fields=fields)

    # Get chronological order — required for temporal split and CV
    batch_ts = get_batch_timestamps(raw)
    features = features.reindex(
        batch_ts.index.intersection(features.index)
    )

    if args.temporal:
        features = add_temporal_features(features, batch_ts)
        features = add_rolling_features(features, batch_ts, window=ROLLING_WINDOW)
        print(f"Added temporal + rolling features -> {features.shape[1]} total features")
    else:
        print(
            f"Feature matrix: {len(features)} batches x {features.shape[1]} features "
            f"({len(fields)} fields x 6 stats)"
        )

    if len(features) < N_CV_SPLITS + 2:
        print(
            f"ERROR: need at least {N_CV_SPLITS + 2} batches to run {N_CV_SPLITS}-fold CV.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Labels -------------------------------------------------------------
    labels = make_health_risk_labels(features)
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    print(f"Labels: {n_pos} positive (health_risk=1), {n_neg} negative (health_risk=0)")
    print(f"Label source: WHO 2021 / ASHRAE 62.1 / EN 15251 thresholds")

    if labels.nunique() < 2:
        print(
            "WARNING: all labels belong to one class. "
            "Classification is trivial - collect more varied conditions."
        )

    # --- Temporal split: first 80% train / last 20% held-out test -----------
    X_all = _impute_nan(features.values.astype(float))
    y_all = labels.values
    feature_names = features.columns.tolist()

    split_idx = int(len(X_all) * 0.8)
    X_train, X_test = X_all[:split_idx], X_all[split_idx:]
    y_train, y_test = y_all[:split_idx], y_all[split_idx:]
    print(f"Temporal split: {len(X_train)} train / {len(X_test)} test (chronological)")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)

    # --- Threshold baseline (deterministic, no scaling needed) --------------
    baseline = _ThresholdBaseline(feature_names)
    baseline_preds_test = baseline.predict(X_test)
    baseline_cv_reports = []
    for _, val_idx in tscv.split(X_train):
        val_preds = baseline.predict(X_train[val_idx])
        baseline_cv_reports.append(
            classification_report_full(y_train[val_idx], val_preds, "threshold_baseline", {
                "latency_ms": 0.0, "latency_per_sample_ms": 0.0,
                "cpu_pct": 0.0, "ram_percent": 0.0, "process_memory_mb": 0.0,
            })
        )

    cv_reports_all = [cv_summary(baseline_cv_reports, "threshold_baseline")]
    final_reports = [
        {
            **classification_report_full(y_test, baseline_preds_test, "threshold_baseline", {
                "latency_ms": 0.0, "latency_per_sample_ms": 0.0,
                "cpu_pct": 0.0, "ram_percent": 0.0, "process_memory_mb": 0.0,
            }),
            "train_time_ms": 0.0, "train_cpu_pct": 0.0,
            "train_ram_percent": 0.0, "train_process_memory_mb": 0.0,
            "model_size_kb": 0.0,
            "split": "temporal_80_20",
        }
    ]
    all_predictions = {"threshold_baseline": baseline_preds_test}

    # --- TimeSeriesSplit CV + final evaluation for each ML model ------------
    print()
    print(f"Running {N_CV_SPLITS}-fold TimeSeriesSplit CV...")
    print()
    print(f"{'Model':<25} {'CV F1':>8} {'CV F1 std':>10} {'Test F1':>8} {'Test Acc':>9} {'Size KB':>8}")
    print("-" * 73)

    for name, clf in _CLASSIFIERS.items():
        # --- CV on training portion -----------------------------------------
        fold_reports = []
        for train_idx, val_idx in tscv.split(X_train):
            Xf_tr, Xf_val = X_train[train_idx], X_train[val_idx]
            yf_tr, yf_val = y_train[train_idx], y_train[val_idx]
            sc = StandardScaler()
            Xf_tr_s  = sc.fit_transform(Xf_tr)
            Xf_val_s = sc.transform(Xf_val)
            train_m = measure_training(clf, Xf_tr_s, yf_tr)
            inf     = measure_inference(clf, Xf_val_s)
            rep     = classification_report_full(yf_val, inf["predictions"], name, inf)
            rep.update(train_m)
            fold_reports.append(rep)

        cv_reports_all.append(cv_summary(fold_reports, name))

        # --- Final model on full train set ----------------------------------
        scaler_final = StandardScaler()
        X_tr_s = scaler_final.fit_transform(X_train)
        X_te_s = scaler_final.transform(X_test)
        train_m   = measure_training(clf, X_tr_s, y_train)
        inf_final = measure_inference(clf, X_te_s)

        final_rep = classification_report_full(y_test, inf_final["predictions"], name, inf_final)
        final_rep.update(train_m)
        final_rep["split"] = "temporal_80_20"

        meta = {
            "task":           "health_risk_classification",
            "label_source":   "WHO2021_ASHRAE_EN15251",
            "feature_fields": fields,
            "feature_names":  feature_names,
            "n_features":     X_all.shape[1],
            "n_train":        len(X_train),
            "n_test":         len(X_test),
            "n_positive":     n_pos,
            "n_negative":     n_neg,
            "temporal_split": True,
            "cv_folds":       N_CV_SPLITS,
            "trained_at":     ts,
        }
        model_path = save_model(clf, name, meta)
        save_model(scaler_final, f"{name}_scaler", {"for_model": name, "trained_at": ts})
        final_rep["model_size_kb"] = round(model_path.stat().st_size / 1024, 2)

        final_reports.append(final_rep)
        all_predictions[name] = inf_final["predictions"]

        cv_f1_mean = cv_reports_all[-1].get("f1_mean", float("nan"))
        cv_f1_std  = cv_reports_all[-1].get("f1_std",  float("nan"))
        print(
            f"{name:<25} {cv_f1_mean:>8.3f} {cv_f1_std:>10.3f} "
            f"{final_rep['f1']:>8.3f} {final_rep['accuracy']:>9.3f} "
            f"{final_rep.get('model_size_kb', 0):>8.1f}"
        )

    # --- McNemar's tests (all pairs) ----------------------------------------
    mcnemar_results = []
    model_names = list(all_predictions.keys())
    for n1, n2 in combinations(model_names, 2):
        mcnemar_results.append(
            mcnemar_test(y_test, all_predictions[n1], all_predictions[n2], n1, n2)
        )

    # --- Save reports -------------------------------------------------------
    jp1, cp1 = save_report(cv_reports_all,   f"classification_cv_{ts}")
    jp2, cp2 = save_report(final_reports,    f"classification_final_{ts}")
    jp3, cp3 = save_report(mcnemar_results,  f"classification_mcnemar_{ts}")

    print()
    print(f"CV report     -> {jp1}")
    print(f"Final report  -> {jp2}")
    print(f"McNemar tests -> {jp3}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train health-risk classification models (temporal CV, WHO 2021 thresholds)"
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument("--input", help="CSV file with raw_observations export")
    src.add_argument("--db",    help=f"DuckDB path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--system",   action="store_true",
                        help="Include cpu_pct and ram_percent as features")
    parser.add_argument("--temporal", action="store_true",
                        help="Add hour_sin/cos, day_of_week, and rolling mean features")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

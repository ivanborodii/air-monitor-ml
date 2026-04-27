"""Train unsupervised anomaly detection models on air-quality micro-batches.

Usage
-----
From the air-monitor/ directory:

    python -m ml.train_anomaly --db data/air_monitor.duckdb
    python -m ml.train_anomaly --input data/microbatches.csv
    python -m ml.train_anomaly --db data/air_monitor.duckdb --temporal

Scientific design
-----------------
- Features sorted chronologically before fitting (consistent with classification).
- LOF+PCA variant added: PCA(n_components=15) reduces 60 features to 15 principal
  components before LOF, addressing the curse of dimensionality that caused
  LOF F1=0.15 on the full 60-dimensional space.
- Two evaluation tracks:
    1. Rule-based ground truth: compare ML models against ANOMALY_RANGE_THRESHOLDS
       labels (shows alignment with domain rules).
    2. Synthetic ground truth: inject anomalies (3-sigma shifts) into 2% of batches
       with known labels, providing model-independent evaluation.
- Rule-based baseline included for reference (F1=1.0 by definition vs its own labels).

Models: IsolationForest, LocalOutlierFactor (full 60-dim),
        LocalOutlierFactor+PCA (15-dim), OneClassSVM.

Reports saved:
  outputs/reports/anomaly_{ts}.json/csv           -- vs rule-based labels
  outputs/reports/anomaly_synthetic_{ts}.json/csv -- vs injected anomalies
"""

import argparse
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from ml.config import CORE_SENSOR_FIELDS, DEFAULT_DB_PATH, OPTIONAL_SYSTEM_FIELDS, ROLLING_WINDOW
from ml.evaluate import anomaly_report_full, save_report
from ml.feature_engineering import (
    add_rolling_features, add_temporal_features,
    build_batch_features, get_batch_timestamps,
    load_from_csv, load_from_duckdb,
)
from ml.labels import inject_synthetic_anomalies, make_anomaly_labels
from ml.metrics import measure_inference, measure_training
from ml.model_registry import save_model

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

_ANOMALY_MODELS = {
    "isolation_forest": IsolationForest(
        n_estimators=50, contamination="auto", random_state=42
    ),
    "local_outlier_factor": LocalOutlierFactor(
        n_neighbors=5, novelty=True, contamination="auto"
    ),
    "local_outlier_factor_pca": Pipeline([
        ("pca", PCA(n_components=15, random_state=42)),
        ("lof", LocalOutlierFactor(n_neighbors=5, novelty=True, contamination="auto")),
    ]),
    "one_class_svm": OneClassSVM(kernel="rbf", nu=0.05),
}


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

    # Sort chronologically (consistent with classification pipeline)
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

    if len(features) < 5:
        print(
            "ERROR: need at least 5 complete batches to train.",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- Rule-based anomaly labels ------------------------------------------
    rule_labels = make_anomaly_labels(features)
    n_rule = int(rule_labels.sum())
    print(f"Rule-based labels: {n_rule} anomalies / {len(rule_labels) - n_rule} normal")

    X = _impute_nan(features.values.astype(float))
    y_rule = rule_labels.values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    feature_names = features.columns.tolist()

    # --- Rule-based baseline ------------------------------------------------
    y_rule_sklearn = np.where(y_rule == 1, -1, 1)
    baseline_inf = {k: 0.0 for k in [
        "latency_ms", "latency_per_sample_ms", "cpu_pct",
        "ram_percent", "process_memory_mb",
    ]}
    baseline_report = anomaly_report_full(y_rule, y_rule_sklearn, "rule_based_baseline", baseline_inf)
    baseline_report.update({
        "train_time_ms": 0.0, "train_cpu_pct": 0.0,
        "train_ram_percent": 0.0, "train_process_memory_mb": 0.0,
    })

    reports = [baseline_report]

    # --- Train and evaluate ML models ---------------------------------------
    print()
    print(f"{'Model':<28} {'Anom':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Train(ms)':>11} {'Infer(ms)':>11}")
    print("-" * 80)
    print(
        f"{'rule_based_baseline':<28} {baseline_report['n_anomalies_detected']:>6d} "
        f"{baseline_report.get('precision', float('nan')):>6.3f} "
        f"{baseline_report.get('recall', float('nan')):>6.3f} "
        f"{baseline_report.get('f1', float('nan')):>6.3f} "
        f"{'(rule)':>11} {'(rule)':>11}"
    )

    trained_models: dict = {}

    for name, model in _ANOMALY_MODELS.items():
        train_m = measure_training(model, X_scaled)
        inf     = measure_inference(model, X_scaled)
        y_pred  = inf["predictions"]
        report  = anomaly_report_full(y_rule, y_pred, name, inf)
        report.update(train_m)
        reports.append(report)
        trained_models[name] = model

        meta = {
            "task":             "anomaly_detection",
            "feature_fields":   fields,
            "feature_names":    feature_names,
            "n_features":       X.shape[1],
            "n_train":          len(X),
            "n_rule_anomalies": n_rule,
            "pca_applied":      "pca" in name,
            "trained_at":       ts,
        }
        save_model(model,  name,             meta)
        save_model(scaler, f"{name}_scaler", {"for_model": name, "trained_at": ts})

        print(
            f"{name:<28} {report['n_anomalies_detected']:>6d} "
            f"{report.get('precision', float('nan')):>6.3f} "
            f"{report.get('recall', float('nan')):>6.3f} "
            f"{report.get('f1', float('nan')):>6.3f} "
            f"{report['train_time_ms']:>11.1f} {report['latency_ms']:>11.2f}"
        )

    # --- Save rule-based evaluation report ----------------------------------
    jp1, cp1 = save_report(reports, f"anomaly_{ts}")
    print()
    print(f"Rule-based report  -> {jp1}")

    # --- Synthetic anomaly evaluation ---------------------------------------
    print()
    print("==> Synthetic anomaly evaluation (3-sigma injections, 2% of batches)...")
    X_synth_feat, y_synth = inject_synthetic_anomalies(features, fraction=0.02, random_state=42)
    n_synth = int(y_synth.sum())
    print(f"    Injected {n_synth} synthetic anomalies into {len(y_synth)} batches")

    X_synth = _impute_nan(X_synth_feat.values.astype(float))
    X_synth_scaled = scaler.transform(X_synth)
    y_synth_arr = y_synth.values

    synth_reports = []
    for name, model in trained_models.items():
        inf    = measure_inference(model, X_synth_scaled)
        y_pred = inf["predictions"]
        rep    = anomaly_report_full(y_synth_arr, y_pred, f"{name}_synth", inf)
        synth_reports.append(rep)
        print(
            f"    {name:<28} detected={rep['n_anomalies_detected']:>4}  "
            f"prec={rep.get('precision', float('nan')):.3f}  "
            f"rec={rep.get('recall', float('nan')):.3f}  "
            f"f1={rep.get('f1', float('nan')):.3f}"
        )

    jp2, cp2 = save_report(synth_reports, f"anomaly_synthetic_{ts}")
    print()
    print(f"Synthetic report   -> {jp2}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train unsupervised anomaly detection models (LOF+PCA, synthetic evaluation)"
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

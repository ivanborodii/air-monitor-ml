"""Run trained models on a new micro-batch and save predictions.

Usage
-----
From the air-monitor/ directory:

    python -m ml.inference --input data/latest_microbatch.json

    # Run only classification or only anomaly models:
    python -m ml.inference --input data/latest_microbatch.json --task classification
    python -m ml.inference --input data/latest_microbatch.json --task anomaly

Input format
------------
A JSON file containing a list of observation dicts — the same structure as
data/batch_buffer.json produced by the pipeline. The batch must contain at
least one observation; missing fields are treated as NaN.

Output
------
outputs/predictions/predictions_{timestamp}.csv and .json
Each row: batch_id, timestamp, model_name, task, prediction_label,
          score, latency_ms, cpu_pct, ram_percent, process_memory_mb
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from ml.config import (
    CORE_SENSOR_FIELDS,
    DEFAULT_DB_PATH,
    OPTIONAL_SYSTEM_FIELDS,
    PREDICTIONS_DIR,
)
from ml.feature_engineering import build_batch_features
from ml.metrics import measure_inference
from ml.model_registry import load_latest_model

_CLASSIFICATION_MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
]

_ANOMALY_MODELS = [
    "isolation_forest",
    "local_outlier_factor",
    "one_class_svm",
]


def _load_batch_json(path: str) -> pd.DataFrame:
    """Load a batch JSON file into a DataFrame, parsing ISO timestamps."""
    with open(path) as fh:
        records = json.load(fh)

    if not records:
        print(f"ERROR: {path} is empty.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(records)
    for col in ("ts", "measured_at", "loaded_at"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    if "batch_id" not in df.columns:
        df["batch_id"] = "inferred_batch"
    if "row_in_batch" not in df.columns:
        df["row_in_batch"] = range(len(df))

    return df


def _impute_nan(X: np.ndarray, col_means: np.ndarray) -> np.ndarray:
    """Impute NaN with provided column means (from training set)."""
    X = X.copy().astype(float)
    nan_idx = np.where(np.isnan(X))
    X[nan_idx] = col_means[nan_idx[1]]
    return X


def _run_classification(X_raw: np.ndarray, batch_id: str, ts: str) -> list[dict]:
    rows = []
    for name in _CLASSIFICATION_MODELS:
        try:
            model,  _  = load_latest_model(name)
            scaler, sm = load_latest_model(f"{name}_scaler")
        except FileNotFoundError as exc:
            print(f"  SKIP {name}: {exc}", file=sys.stderr)
            continue

        col_means = np.zeros(X_raw.shape[1])
        X = _impute_nan(X_raw, col_means)
        X_scaled = scaler.transform(X)

        inf = measure_inference(model, X_scaled)
        pred = int(inf["predictions"][0])
        score = float(inf["scores"][0]) if inf["scores"] is not None else None
        label = "health_risk" if pred == 1 else "no_risk"

        rows.append({
            "batch_id":           batch_id,
            "timestamp":          ts,
            "task":               "classification",
            "model_name":         name,
            "prediction":         pred,
            "prediction_label":   label,
            "score":              score,
            "latency_ms":         inf["latency_ms"],
            "cpu_pct":            inf["cpu_pct"],
            "ram_percent":        inf["ram_percent"],
            "process_memory_mb":  inf["process_memory_mb"],
        })
        print(
            f"  {name:<25}  {label:<12}  "
            f"score={score:.4f if score is not None else 'n/a':<8}  "
            f"lat={inf['latency_ms']:.2f}ms"
        )
    return rows


def _run_anomaly(X_raw: np.ndarray, batch_id: str, ts: str) -> list[dict]:
    rows = []
    for name in _ANOMALY_MODELS:
        try:
            model,  _  = load_latest_model(name)
            scaler, _  = load_latest_model(f"{name}_scaler")
        except FileNotFoundError as exc:
            print(f"  SKIP {name}: {exc}", file=sys.stderr)
            continue

        col_means = np.zeros(X_raw.shape[1])
        X = _impute_nan(X_raw, col_means)
        X_scaled = scaler.transform(X)

        inf = measure_inference(model, X_scaled)
        pred_sklearn = int(inf["predictions"][0])   # -1 or +1
        score = float(inf["scores"][0]) if inf["scores"] is not None else None
        label = "anomaly" if pred_sklearn == -1 else "normal"

        rows.append({
            "batch_id":           batch_id,
            "timestamp":          ts,
            "task":               "anomaly_detection",
            "model_name":         name,
            "prediction":         pred_sklearn,
            "prediction_label":   label,
            "score":              score,
            "latency_ms":         inf["latency_ms"],
            "cpu_pct":            inf["cpu_pct"],
            "ram_percent":        inf["ram_percent"],
            "process_memory_mb":  inf["process_memory_mb"],
        })
        print(
            f"  {name:<25}  {label:<12}  "
            f"score={score if score is not None else 'n/a':<12}  "
            f"lat={inf['latency_ms']:.2f}ms"
        )
    return rows


def run(args: argparse.Namespace) -> None:
    # --- Load batch -------------------------------------------------------
    df = _load_batch_json(args.input)
    batch_id = str(df["batch_id"].iloc[0]) if "batch_id" in df.columns else "unknown"
    ts = datetime.now(timezone.utc).isoformat()

    print(f"Batch: {batch_id}  ({len(df)} observations)")

    # --- Build features ---------------------------------------------------
    fields = list(CORE_SENSOR_FIELDS)
    features = build_batch_features(df, fields=fields)

    if features.empty:
        print("ERROR: could not build features - batch may be entirely NaN.", file=sys.stderr)
        sys.exit(1)

    X_raw = features.values.astype(float)

    # --- Inference --------------------------------------------------------
    all_rows: list[dict] = []

    if args.task in ("classification", "all"):
        print("\n--- Classification ---")
        all_rows += _run_classification(X_raw, batch_id, ts)

    if args.task in ("anomaly", "all"):
        print("\n--- Anomaly Detection ---")
        all_rows += _run_anomaly(X_raw, batch_id, ts)

    if not all_rows:
        print("No predictions generated - train models first.", file=sys.stderr)
        sys.exit(1)

    # --- Save predictions -------------------------------------------------
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    out_ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base     = os.path.join(PREDICTIONS_DIR, f"predictions_{out_ts}")
    json_path = f"{base}.json"
    csv_path  = f"{base}.csv"

    with open(json_path, "w") as fh:
        json.dump(all_rows, fh, indent=2, default=str)

    keys = list(all_rows[0].keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nPredictions -> {json_path}")
    print(f"            -> {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained ML models on a new air-quality micro-batch"
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to micro-batch JSON file (e.g. data/latest_microbatch.json)"
    )
    parser.add_argument(
        "--task",
        choices=["classification", "anomaly", "all"],
        default="all",
        help="Which models to run (default: all)",
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()

"""Compute evaluation metrics and save reports to disk.

classification_report_full() — accuracy, precision, recall, F1, confusion
matrix, plus inference resource metrics.

anomaly_report_full() — precision, recall, F1 against rule-based ground
truth, plus counts and inference resource metrics.

save_report() — writes both JSON and CSV for reproducible comparison.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from ml.config import REPORTS_DIR


def classification_report_full(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    inference_metrics: dict,
    label_name: str = "health_risk",
) -> dict:
    """Build a flat evaluation dict for one classification model."""
    return {
        "model":              model_name,
        "label":              label_name,
        "n_samples":          int(len(y_true)),
        "n_positive":         int(y_true.sum()),
        "n_negative":         int((y_true == 0).sum()),
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "precision":          float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":             float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":                 float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix":   confusion_matrix(y_true, y_pred).tolist(),
        "latency_ms":         inference_metrics["latency_ms"],
        "latency_per_sample_ms": inference_metrics["latency_per_sample_ms"],
        "cpu_pct":            inference_metrics["cpu_pct"],
        "ram_percent":        inference_metrics["ram_percent"],
        "process_memory_mb":  inference_metrics["process_memory_mb"],
    }


def anomaly_report_full(
    y_true: Optional[np.ndarray],
    y_pred_sklearn: np.ndarray,
    model_name: str,
    inference_metrics: dict,
) -> dict:
    """Build a flat evaluation dict for one anomaly detection model.

    y_pred_sklearn uses sklearn convention: -1 = anomaly, +1 = normal.
    y_true uses binary convention: 1 = anomaly, 0 = normal.
    """
    # Convert sklearn convention to binary
    y_pred_binary = (y_pred_sklearn == -1).astype(int)
    n_detected = int(y_pred_binary.sum())

    report: dict = {
        "model":              model_name,
        "n_samples":          int(len(y_pred_sklearn)),
        "n_anomalies_detected": n_detected,
        "latency_ms":         inference_metrics["latency_ms"],
        "latency_per_sample_ms": inference_metrics["latency_per_sample_ms"],
        "cpu_pct":            inference_metrics["cpu_pct"],
        "ram_percent":        inference_metrics["ram_percent"],
        "process_memory_mb":  inference_metrics["process_memory_mb"],
    }

    if y_true is not None and len(y_true) == len(y_pred_binary):
        report["n_true_anomalies"] = int(y_true.sum())
        report["precision"] = float(precision_score(y_true, y_pred_binary, zero_division=0))
        report["recall"]    = float(recall_score(y_true, y_pred_binary, zero_division=0))
        report["f1"]        = float(f1_score(y_true, y_pred_binary, zero_division=0))

    return report


def save_report(reports: list[dict], name: str) -> tuple[str, str]:
    """Persist a list of report dicts as {name}.json and {name}.csv.

    Returns (json_path, csv_path).
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    base      = os.path.join(REPORTS_DIR, name)
    json_path = f"{base}.json"
    csv_path  = f"{base}.csv"

    with open(json_path, "w") as fh:
        json.dump(reports, fh, indent=2, default=str)

    if reports:
        keys = list(dict.fromkeys(k for r in reports for k in r.keys()))
        with open(csv_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore", restval="")
            writer.writeheader()
            for row in reports:
                flat = {
                    k: (str(v) if isinstance(v, (list, dict)) else v)
                    for k, v in row.items()
                }
                writer.writerow(flat)

    return json_path, csv_path


def cv_summary(fold_reports: list[dict], model_name: str) -> dict:
    """Aggregate per-fold classification metrics into mean and std.

    fold_reports: list of dicts from classification_report_full(), one per CV fold.
    Returns a flat dict suitable for save_report().
    """
    keys = [
        "accuracy", "precision", "recall", "f1",
        "latency_ms", "train_time_ms", "cpu_pct", "ram_percent",
    ]
    summary: dict = {"model": model_name, "n_folds": len(fold_reports)}
    for k in keys:
        vals = [r[k] for r in fold_reports if k in r]
        if vals:
            summary[f"{k}_mean"] = float(np.mean(vals))
            summary[f"{k}_std"]  = float(np.std(vals))
    return summary


def mcnemar_test(
    y_true: np.ndarray,
    y_pred1: np.ndarray,
    y_pred2: np.ndarray,
    model1_name: str,
    model2_name: str,
) -> dict:
    """McNemar's test with continuity correction for paired classifier comparison.

    Tests whether two classifiers make statistically different errors on the
    same samples. p_value < 0.05 indicates a significant performance difference.

    b = cases correct by model1 but wrong by model2
    c = cases wrong by model1 but correct by model2
    Uses scipy.stats.chi2 (available as a scikit-learn transitive dependency).
    """
    from scipy.stats import chi2 as _chi2
    b = int(((y_pred1 == y_true) & (y_pred2 != y_true)).sum())
    c = int(((y_pred1 != y_true) & (y_pred2 == y_true)).sum())
    n = b + c
    if n == 0:
        return {
            "model1": model1_name, "model2": model2_name,
            "b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0, "significant": False,
        }
    statistic = float((abs(b - c) - 1) ** 2 / n)
    p_value   = float(1.0 - _chi2.cdf(statistic, df=1))
    return {
        "model1":     model1_name,
        "model2":     model2_name,
        "b":          b,
        "c":          c,
        "statistic":  statistic,
        "p_value":    p_value,
        "significant": p_value < 0.05,
    }

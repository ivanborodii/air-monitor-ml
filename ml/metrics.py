"""Measure CPU, RAM, and wall-clock latency during model inference.

measure_inference() wraps a single model.predict() call and returns both
the predictions and the resource snapshot. Confidence scores are extracted
via predict_proba / decision_function / score_samples when available.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import numpy as np
import psutil


def measure_training(model: Any, X: np.ndarray, y: np.ndarray | None = None) -> dict:
    """Run model.fit() and capture resource usage around the call.

    Pass y for supervised models; omit it for unsupervised (IsolationForest, etc.).

    Returns
    -------
    dict with keys:
        train_time_ms           — wall-clock time for fit() [ms]
        train_cpu_pct           — average CPU % during fit
        train_ram_percent       — average RAM % during fit
        train_process_memory_mb — average process RSS [MB]
    """
    process = psutil.Process()

    cpu_before = psutil.cpu_percent(interval=None)
    ram_before = psutil.virtual_memory().percent
    mem_before = process.memory_info().rss / 1024 / 1024

    t0 = time.perf_counter()
    if y is not None:
        model.fit(X, y)
    else:
        model.fit(X)
    train_time_ms = (time.perf_counter() - t0) * 1000.0

    cpu_after = psutil.cpu_percent(interval=None)
    ram_after = psutil.virtual_memory().percent
    mem_after = process.memory_info().rss / 1024 / 1024

    return {
        "train_time_ms":           train_time_ms,
        "train_cpu_pct":           (cpu_before + cpu_after) / 2.0,
        "train_ram_percent":       (ram_before + ram_after) / 2.0,
        "train_process_memory_mb": (mem_before + mem_after) / 2.0,
    }


def measure_inference(model: Any, X: np.ndarray) -> dict:
    """Run model.predict(X) and capture resource usage around the call.

    Returns
    -------
    dict with keys:
        predictions          — raw output of model.predict(X)
        scores               — confidence / anomaly scores (None if unavailable)
        latency_ms           — total wall-clock time for predict() [ms]
        latency_per_sample_ms — latency_ms / n_samples [ms]
        cpu_pct              — average CPU % (sampled before + after)
        ram_percent          — average RAM % (sampled before + after)
        process_memory_mb    — average process RSS [MB]
    """
    process = psutil.Process()

    cpu_before  = psutil.cpu_percent(interval=None)
    ram_before  = psutil.virtual_memory().percent
    mem_before  = process.memory_info().rss / 1024 / 1024

    t0 = time.perf_counter()
    predictions = model.predict(X)
    latency_ms  = (time.perf_counter() - t0) * 1000.0

    cpu_after   = psutil.cpu_percent(interval=None)
    ram_after   = psutil.virtual_memory().percent
    mem_after   = process.memory_info().rss / 1024 / 1024

    n_samples = max(len(X), 1)

    scores: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        try:
            proba  = model.predict_proba(X)
            scores = proba.max(axis=1)
        except Exception:
            pass
    if scores is None and hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X)
        except Exception:
            pass
    if scores is None and hasattr(model, "score_samples"):
        try:
            scores = model.score_samples(X)
        except Exception:
            pass

    return {
        "predictions":           predictions,
        "scores":                scores,
        "latency_ms":            latency_ms,
        "latency_per_sample_ms": latency_ms / n_samples,
        "cpu_pct":               (cpu_before + cpu_after) / 2.0,
        "ram_percent":           (ram_before + ram_after) / 2.0,
        "process_memory_mb":     (mem_before + mem_after) / 2.0,
    }

"""Generate binary labels for supervised and unsupervised tasks.

health_risk  — 1 if any configured threshold is exceeded in the batch.
anomaly      — 1 if the intra-batch range of any signal exceeds its threshold.

Both functions consume the feature DataFrame produced by
feature_engineering.build_batch_features(), so they operate on aggregated
batch-level statistics (mean, min, max, range, …) rather than raw rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.config import HEALTH_THRESHOLDS, ANOMALY_RANGE_THRESHOLDS


def make_health_risk_labels(
    features: pd.DataFrame,
    thresholds: dict | None = None,
) -> pd.Series:
    """Return a binary Series (0 = safe, 1 = risk) per batch.

    Rules:
      - For each field with a "max" bound: flag if {field}_max > bound.
      - For each field with a "min" bound: flag if {field}_min < bound.
    A batch is flagged if ANY rule fires.
    """
    if thresholds is None:
        thresholds = HEALTH_THRESHOLDS

    risk = pd.Series(0, index=features.index, dtype=int)

    for field, bounds in thresholds.items():
        if "max" in bounds:
            col = f"{field}_max"
            if col in features.columns:
                risk |= (features[col] > bounds["max"]).fillna(False).astype(int)
        if "min" in bounds:
            col = f"{field}_min"
            if col in features.columns:
                risk |= (features[col] < bounds["min"]).fillna(False).astype(int)

    return risk.rename("health_risk")


def make_anomaly_labels(
    features: pd.DataFrame,
    thresholds: dict | None = None,
) -> pd.Series:
    """Return a binary Series (0 = normal, 1 = anomaly) per batch.

    Each key in thresholds is a feature column name ending in _range.
    A batch is flagged if ANY range exceeds its threshold.
    """
    if thresholds is None:
        thresholds = ANOMALY_RANGE_THRESHOLDS

    anomaly = pd.Series(0, index=features.index, dtype=int)

    for col, threshold in thresholds.items():
        if col in features.columns:
            anomaly |= (features[col] > threshold).fillna(False).astype(int)

    return anomaly.rename("anomaly")


def inject_synthetic_anomalies(
    features: pd.DataFrame,
    fraction: float = 0.02,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Inject synthetic anomalies by shifting range features by 3 standard deviations.

    Produces ground-truth labels that are independent of the ML model,
    breaking the evaluation circularity of comparing against rule-based labels
    derived from the same feature statistics.

    Parameters
    ----------
    features:   Feature DataFrame from build_batch_features().
    fraction:   Fraction of batches to corrupt (default 2%).
    random_state: Seed for reproducibility.

    Returns
    -------
    (modified_features_copy, binary_labels)  where 1 = injected anomaly.
    """
    rng = np.random.default_rng(random_state)
    n_inject = max(1, int(len(features) * fraction))
    injected = features.copy()
    range_cols = [c for c in features.columns if c.endswith("_range")]

    if not range_cols:
        raise ValueError("No _range columns found in features.")

    chosen_idx = rng.choice(len(features), size=n_inject, replace=False)
    for i in chosen_idx:
        col = str(rng.choice(range_cols))
        std = float(features[col].std())
        if std > 0:
            injected.iat[i, features.columns.get_loc(col)] += 3.0 * std

    labels = pd.Series(0, index=features.index, dtype=int, name="synthetic_anomaly")
    labels.iloc[chosen_idx] = 1
    return injected, labels

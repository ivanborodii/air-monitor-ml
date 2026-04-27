"""Build per-batch feature vectors from raw observation rows.

Each micro-batch of MICROBATCH_SIZE observations is reduced to a single
feature vector by computing six statistics per sensor field:
    mean, min, max, std, delta (last − first), range (max − min)

The resulting DataFrame has one row per batch_id and
len(fields) × 6 columns named {field}_{stat}.
"""

from __future__ import annotations

import warnings
from typing import Optional

import duckdb
import numpy as np
import pandas as pd

from ml.config import CORE_SENSOR_FIELDS, OPTIONAL_SYSTEM_FIELDS, DEFAULT_DB_PATH

_STATS = ("mean", "min", "max", "std", "delta", "range")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_duckdb(
    db_path: str = DEFAULT_DB_PATH,
    include_system: bool = False,
) -> pd.DataFrame:
    """Load raw_observations from DuckDB, optionally joined with runtime_metrics."""
    con = duckdb.connect(db_path, read_only=True)
    try:
        if include_system:
            query = """
                SELECT
                    r.batch_id, r.row_in_batch, r.ts, r.measured_at,
                    r.mass_pm1_0, r.mass_pm2_5, r.mass_pm4_0, r.mass_pm10,
                    r.number_pm0_5, r.number_pm1_0, r.number_pm2_5,
                    r.number_pm4_0, r.number_pm10, r.typical_size_um,
                    r.co2_ppm, r.scd_temp_c, r.scd_humidity_pct,
                    r.bme_temp_c, r.bme_humidity_pct, r.pressure_hpa,
                    r.gas_resistance_ohm,
                    m.cpu_pct, m.ram_percent
                FROM raw_observations r
                LEFT JOIN runtime_metrics m
                    ON r.batch_id = m.batch_id AND r.row_in_batch = m.row_in_batch
                ORDER BY r.batch_id, r.row_in_batch
            """
        else:
            query = """
                SELECT
                    batch_id, row_in_batch, ts, measured_at,
                    mass_pm1_0, mass_pm2_5, mass_pm4_0, mass_pm10,
                    number_pm0_5, number_pm1_0, number_pm2_5,
                    number_pm4_0, number_pm10, typical_size_um,
                    co2_ppm, scd_temp_c, scd_humidity_pct,
                    bme_temp_c, bme_humidity_pct, pressure_hpa,
                    gas_resistance_ohm
                FROM raw_observations
                ORDER BY batch_id, row_in_batch
            """
        df = con.execute(query).df()
    finally:
        con.close()
    return df


def load_from_csv(path: str) -> pd.DataFrame:
    """Load a raw_observations CSV export."""
    return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _batch_stats(group: pd.DataFrame, fields: list[str]) -> pd.Series:
    """Compute 6 statistics per field for one micro-batch group."""
    group = group.sort_values("row_in_batch")
    out: dict = {}
    for f in fields:
        if f not in group.columns:
            for stat in _STATS:
                out[f"{f}_{stat}"] = np.nan
            continue

        valid: pd.Series = group[f].dropna()
        n = len(valid)

        if n == 0:
            for stat in _STATS:
                out[f"{f}_{stat}"] = np.nan
            continue

        out[f"{f}_mean"]  = float(valid.mean())
        out[f"{f}_min"]   = float(valid.min())
        out[f"{f}_max"]   = float(valid.max())
        out[f"{f}_std"]   = float(valid.std()) if n > 1 else 0.0
        out[f"{f}_delta"] = float(valid.iloc[-1] - valid.iloc[0]) if n >= 2 else 0.0
        out[f"{f}_range"] = float(valid.max() - valid.min())

    return pd.Series(out)


def build_batch_features(
    df: pd.DataFrame,
    fields: Optional[list[str]] = None,
    include_system: bool = False,
    nan_drop_threshold: float = 0.5,
) -> pd.DataFrame:
    """Aggregate raw observations into one feature row per batch_id.

    Parameters
    ----------
    df:
        DataFrame with raw_observations rows (one row per sensor reading).
        Must contain columns 'batch_id' and 'row_in_batch'.
    fields:
        Sensor fields to aggregate. Defaults to CORE_SENSOR_FIELDS
        (+ OPTIONAL_SYSTEM_FIELDS when include_system=True).
    include_system:
        When True, append OPTIONAL_SYSTEM_FIELDS to the field list.
    nan_drop_threshold:
        Drop a batch if more than this fraction of its feature values are NaN.

    Returns
    -------
    pd.DataFrame indexed by batch_id, shape (n_batches, n_fields × 6).
    """
    if fields is None:
        fields = list(CORE_SENSOR_FIELDS)
        if include_system:
            fields += OPTIONAL_SYSTEM_FIELDS

    if "batch_id" not in df.columns:
        df = df.copy()
        df["batch_id"] = "batch_0"
    if "row_in_batch" not in df.columns:
        df = df.copy()
        df["row_in_batch"] = range(len(df))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        features = (
            df.groupby("batch_id", sort=True)
            .apply(_batch_stats, fields=fields)
        )

    n_before = len(features)
    nan_frac = features.isna().mean(axis=1)
    features = features[nan_frac <= nan_drop_threshold]
    n_dropped = n_before - len(features)
    if n_dropped > 0:
        warnings.warn(
            f"Dropped {n_dropped} batch(es) where >{nan_drop_threshold*100:.0f}% "
            "of features were NaN (likely failed sensors during that batch).",
            UserWarning,
            stacklevel=2,
        )

    return features


# ---------------------------------------------------------------------------
# Temporal ordering and enrichment
# ---------------------------------------------------------------------------

def get_batch_timestamps(df: pd.DataFrame) -> pd.Series:
    """Return Series batch_id -> earliest ts, sorted chronologically.

    Handles both DuckDB (datetime) and CSV (string) inputs by coercing ts to
    UTC-aware datetime. Used to sort feature matrix before temporal splits.
    """
    ts_col = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df.assign(ts=ts_col).groupby("batch_id")["ts"].min().sort_values()


def add_temporal_features(
    features: pd.DataFrame,
    batch_ts: pd.Series,
) -> pd.DataFrame:
    """Add hour_sin, hour_cos, day_of_week derived from batch timestamp.

    Cyclic sine/cosine encoding of hour prevents the model treating
    23:00 and 00:00 as far apart. day_of_week captures weekly patterns.
    """
    ts_aligned = pd.to_datetime(
        batch_ts.reindex(features.index), utc=True, errors="coerce"
    )
    hour = ts_aligned.dt.hour.fillna(0)
    features = features.copy()
    features["hour_sin"]    = np.sin(2 * np.pi * hour / 24).astype(float)
    features["hour_cos"]    = np.cos(2 * np.pi * hour / 24).astype(float)
    features["day_of_week"] = ts_aligned.dt.dayofweek.fillna(0).astype(float)
    return features


def add_rolling_features(
    features: pd.DataFrame,
    batch_ts: pd.Series,
    window: int = 3,
    fields: list[str] | None = None,
) -> pd.DataFrame:
    """Add N-batch rolling mean for key sensor mean features.

    Captures multi-batch trends (e.g. gradual CO2 build-up across batches).
    Batches are sorted chronologically, rolling applied, then reindexed to
    the original order. NaN for the first (window-1) batches is later imputed.
    """
    if fields is None:
        fields = [
            "co2_ppm_mean", "mass_pm2_5_mean", "scd_temp_c_mean",
            "scd_humidity_pct_mean", "pressure_hpa_mean",
        ]
    sorted_idx = batch_ts.reindex(features.index).sort_values().index
    feat_sorted = features.loc[sorted_idx].copy()
    for f in fields:
        if f in feat_sorted.columns:
            feat_sorted[f"{f}_roll{window}"] = (
                feat_sorted[f].rolling(window, min_periods=1).mean()
            )
    return feat_sorted.reindex(features.index)

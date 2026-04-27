"""Central configuration for the ML module.

All thresholds, field lists, and output paths are defined here so that
train_classification.py, train_anomaly.py, labels.py, and inference.py
never hard-code domain knowledge.
"""

from config.constants import MICROBATCH_SIZE  # noqa: F401 — re-exported for convenience

# ---------------------------------------------------------------------------
# Sensor fields used for feature engineering
# ---------------------------------------------------------------------------

# gas_resistance_ohm is excluded: the BME688 reads a fixed ADC-saturated
# constant (102400000 Ω) on this device, so it carries zero variance.
CORE_SENSOR_FIELDS: list[str] = [
    "mass_pm1_0",
    "mass_pm2_5",
    "mass_pm4_0",
    "mass_pm10",
    "co2_ppm",
    "scd_temp_c",
    "scd_humidity_pct",
    "bme_temp_c",
    "bme_humidity_pct",
    "pressure_hpa",
]

# power_w and cumulative_energy_wh are excluded: always 0.0 on this device
# (power_source='not_configured').
OPTIONAL_SYSTEM_FIELDS: list[str] = [
    "cpu_pct",
    "ram_percent",
]

# ---------------------------------------------------------------------------
# Health risk classification thresholds
# Keys are raw sensor field names (not feature names).
# "max": value exceeded → risk; "min": value below → risk.
# Applied to the per-batch max (for "max" rules) and min (for "min" rules)
# aggregated features.
# ---------------------------------------------------------------------------

# Sources:
#   WHO 2021 Global Air Quality Guidelines (PM2.5, PM10)
#   ASHRAE 62.1-2022 (CO2 indoor comfort limit)
#   EN ISO 7730 / EN 15251 (temperature comfort range)
#   ASHRAE Standard 55-2020 (humidity acceptable range)
HEALTH_THRESHOLDS: dict = {
    "mass_pm2_5":        {"max": 15.0},   # WHO 2021: 15 ug/m3 24h mean
    "mass_pm10":         {"max": 45.0},   # WHO 2021: 45 ug/m3 24h mean
    "co2_ppm":           {"max": 1000.0}, # ASHRAE 62.1: 1000 ppm indoor limit
    "scd_temp_c":        {"min": 19.0, "max": 24.0},  # EN 15251 comfort range
    "scd_humidity_pct":  {"min": 30.0, "max": 70.0},  # ASHRAE 55 acceptable range
}

# ---------------------------------------------------------------------------
# Rule-based anomaly detection thresholds
# Keys are feature names (field + _range suffix) as produced by
# feature_engineering.build_batch_features().
# ---------------------------------------------------------------------------

ANOMALY_RANGE_THRESHOLDS: dict = {
    "mass_pm2_5_range":       15.0,   # µg/m³ swing within one micro-batch
    "co2_ppm_range":          200.0,  # ppm swing within one micro-batch
    "scd_temp_c_range":       1.5,    # °C swing within one micro-batch
    "scd_humidity_pct_range": 10.0,   # % RH swing within one micro-batch
    "pressure_hpa_range":     2.0,    # hPa swing within one micro-batch
}

# ---------------------------------------------------------------------------
# Output paths (relative to air-monitor/ project root)
# ---------------------------------------------------------------------------

MODELS_DIR: str      = "outputs/models"
REPORTS_DIR: str     = "outputs/reports"
PREDICTIONS_DIR: str = "outputs/predictions"

DEFAULT_DB_PATH: str = "data/air_monitor.duckdb"

# ---------------------------------------------------------------------------
# Cross-validation and feature enrichment
# ---------------------------------------------------------------------------

N_CV_SPLITS: int   = 5  # TimeSeriesSplit folds (temporal cross-validation)
ROLLING_WINDOW: int = 3  # batches to look back for rolling mean features

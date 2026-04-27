# ML Module — Air Quality Edge AI (Raspberry Pi 5)

Local machine-learning pipeline for health-risk classification and anomaly
detection on sensor micro-batches. No cloud services, no deep learning.
Designed for reproducible scientific evaluation on a Raspberry Pi 5.

---

## ML Tasks

### Task 1 — Health Risk Classification (supervised)

Binary label: `health_risk = 1` if any threshold is exceeded in the batch.

| Rule | Threshold |
|------|-----------|
| PM2.5 max | > 25 µg/m³ |
| PM10 max | > 50 µg/m³ |
| CO2 max | > 1000 ppm |
| Temperature min/max | < 19 °C or > 24 °C |
| Humidity min/max | < 30 % or > 70 % |

Models: `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`

### Task 2 — Micro-Batch Anomaly Detection (unsupervised)

Flags batches where sensor readings change abnormally fast within a single
micro-batch (sharp intra-batch swings).

| Rule | Threshold |
|------|-----------|
| PM2.5 range | > 15 µg/m³ |
| CO2 range | > 200 ppm |
| Temperature range | > 1.5 °C |
| Humidity range | > 10 % |
| Pressure range | > 2 hPa |

Models: `IsolationForest`, `LocalOutlierFactor` (novelty=True), `OneClassSVM`

Rule-based labels are also generated as a deterministic baseline and used as
ground truth for precision / recall / F1 evaluation.

---

## Feature Generation

Each micro-batch of 10 observations is reduced to a single feature vector.

**Sensor fields aggregated (10 fields):**
- SPS30: `mass_pm1_0`, `mass_pm2_5`, `mass_pm4_0`, `mass_pm10`
- SCD41: `co2_ppm`, `scd_temp_c`, `scd_humidity_pct`
- BME688: `bme_temp_c`, `bme_humidity_pct`, `pressure_hpa`

> `gas_resistance_ohm` is excluded: reads a fixed ADC-saturated constant on
> this device (102400000 Ω, zero variance).

**Six statistics per field:**

| Statistic | Description |
|-----------|-------------|
| `mean`    | Arithmetic mean of valid readings |
| `min`     | Minimum |
| `max`     | Maximum |
| `std`     | Standard deviation (0 if only one reading) |
| `delta`   | Last valid − first valid (temporal trend) |
| `range`   | max − min (intra-batch variability) |

Total: **10 fields × 6 stats = 60 features** (+ 12 optional system features
when `--system` is passed).

---

## Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1-score
- Confusion matrix
- Inference latency (ms), CPU %, RAM %

### Anomaly Detection
- Precision, Recall, F1-score vs rule-based ground truth
- Number of detected anomalies
- Inference latency (ms), CPU %, RAM %

> Power and energy metrics are omitted because the INA219 meter is not
> connected on this device (`power_source = not_configured`).

---

## Setup

Install ML dependencies (from `air-monitor/`):

```bash
pip install scikit-learn>=1.4.0 pandas>=2.0.0 numpy>=1.26.0 joblib>=1.3.0
# or:
pip install -r requirements.txt
```

---

## How to Run

All commands are run from the `air-monitor/` directory.

### 1. Export data from DuckDB (optional, creates a CSV)

```bash
python scripts/export_microbatches.py
# → data/microbatches.csv
```

### 2. Train classification models

```bash
# From DuckDB directly (recommended):
python -m ml.train_classification --db data/air_monitor.duckdb

# From a CSV export:
python -m ml.train_classification --input data/microbatches.csv

# Also include cpu_pct / ram_percent as features:
python -m ml.train_classification --db data/air_monitor.duckdb --system
```

### 3. Train anomaly detection models

```bash
python -m ml.train_anomaly --db data/air_monitor.duckdb
python -m ml.train_anomaly --input data/microbatches.csv
```

### 4. Run inference on a new micro-batch

```bash
# Copy the current JSON buffer as the input batch:
cp data/batch_buffer.json data/latest_microbatch.json

python -m ml.inference --input data/latest_microbatch.json
# Runs both classification and anomaly models by default.

# Only classification:
python -m ml.inference --input data/latest_microbatch.json --task classification

# Only anomaly detection:
python -m ml.inference --input data/latest_microbatch.json --task anomaly
```

---

## Output Files

```
outputs/
├── models/
│   ├── logistic_regression_20240101T000000Z.pkl
│   ├── logistic_regression_20240101T000000Z_meta.json
│   ├── logistic_regression_scaler_20240101T000000Z.pkl
│   ├── decision_tree_*.pkl / *_meta.json / *_scaler*.pkl
│   ├── random_forest_*.pkl / ...
│   ├── isolation_forest_*.pkl / ...
│   ├── local_outlier_factor_*.pkl / ...
│   └── one_class_svm_*.pkl / ...
├── reports/
│   ├── classification_20240101T000000Z.json
│   ├── classification_20240101T000000Z.csv
│   ├── anomaly_20240101T000000Z.json
│   └── anomaly_20240101T000000Z.csv
└── predictions/
    ├── predictions_20240101T000000Z.json
    └── predictions_20240101T000000Z.csv
```

**Prediction row columns:**

| Column | Description |
|--------|-------------|
| `batch_id` | UUID of the micro-batch |
| `timestamp` | ISO 8601 inference time |
| `task` | `classification` or `anomaly_detection` |
| `model_name` | Name of the model |
| `prediction` | Raw model output (0/1 or -1/+1) |
| `prediction_label` | Human-readable: `no_risk`/`health_risk` or `normal`/`anomaly` |
| `score` | Confidence (predict_proba), decision score, or anomaly score |
| `latency_ms` | Inference wall-clock time [ms] |
| `cpu_pct` | CPU usage during inference [%] |
| `ram_percent` | RAM usage during inference [%] |
| `process_memory_mb` | Process RSS during inference [MB] |

---

## Configuration

All thresholds and field lists are in `ml/config.py`. Edit that file to
adjust classification rules, anomaly sensitivity, or which sensor fields
to include in features — no code changes needed elsewhere.

---

## Model Selection Rationale

| Model | Reason |
|-------|--------|
| LogisticRegression | Fast, interpretable baseline; suitable for linearly separable risk conditions |
| DecisionTreeClassifier | Captures threshold-based rules; directly interpretable (depth=5) |
| RandomForestClassifier | Best generalisation in small-data regimes; n_estimators=50 for RPi5 speed |
| IsolationForest | Efficient tree-based anomaly detection; no distance computation |
| LocalOutlierFactor | Density-based; novelty=True enables predict() on new batches |
| OneClassSVM | Kernel-based boundary; effective in low-dimensional feature spaces |

All models use `class_weight="balanced"` (classifiers) or `contamination="auto"` /
`nu=0.05` (anomaly) to handle class imbalance. Models are saved with
`joblib.dump(compress=3)` to minimise storage on SD card.

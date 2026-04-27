# air-monitor

Local edge air quality monitoring pipeline for Raspberry Pi 5.

Continuously reads three sensors, groups observations into configurable
micro-batches, and stores all data in a local DuckDB database.

---

## Project purpose

Collect synchronized air quality observations from three I2C sensors:

| Sensor | Measurements |
|--------|-------------|
| SCD41  | CO2 (ppm), temperature (°C), humidity (%) |
| BME688 | temperature (°C), humidity (%), pressure (hPa), gas resistance (Ω) |
| SPS30  | PM1.0 / PM2.5 / PM4.0 / PM10 mass and number concentrations, typical particle size |

Every observation is linked to a micro-batch. Per-cycle runtime metrics
(sensor timings, system resources, optional power) are stored alongside
the sensor data for later analysis.

---

## Project tree

```
air-monitor/
├── config/
│   ├── constants.py        ← MICROBATCH_SIZE and hardware constants
│   ├── settings.yaml       ← runtime configuration
│   └── thresholds.yaml     ← alert thresholds (future labelling)
├── data/
│   ├── air_monitor.duckdb  ← created by init_db.py
│   └── logs/               ← rotating pipeline.log
├── sensors/
│   ├── scd41.py            ← SCD41Sensor (smbus2, raw I2C)
│   ├── bme688.py           ← BME688Sensor (bme680 library)
│   ├── sps30.py            ← SPS30Sensor (Sensirion driver)
│   └── sync.py             ← fault-isolated coordinated read
├── storage/
│   ├── schema.sql          ← DuckDB table definitions
│   └── duckdb_client.py    ← DuckDBClient (single connection)
├── pipeline/
│   ├── batching.py         ← BatchManager (crash recovery)
│   ├── collector.py        ← PipelineCollector.run_cycle()
│   ├── metrics.py          ← psutil system metrics
│   └── power.py            ← PowerMeter interface (null impl)
├── scripts/
│   ├── init_db.py          ← one-time schema initialisation
│   └── run_pipeline.py     ← main entry point
├── requirements.txt
└── README.md
```

---

## Setup on Raspberry Pi 5

### Prerequisites

```bash
# Enable I2C in raspi-config if not already done
sudo raspi-config nonint do_i2c 0

# Verify sensors are visible on the I2C bus
i2cdetect -y 1
# Expected addresses: 0x62 (SCD41), 0x69 (SPS30), 0x77 (BME688)
```

### Virtual environment

```bash
cd /home/ivan/python_scripts/air_ml
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
cd air-monitor
pip install -r requirements.txt
```

The three new packages needed are `duckdb`, `psutil`, and `PyYAML`.
All sensor libraries should already be present.

### Sensor library notes

**SCD41**: Uses `smbus2` for raw I2C communication — no dedicated driver required.
I2C address: `0x62`.

**BME688**: Uses the `bme680` Python library (v2.0.0). The BME688 is
pin-compatible with the BME680 and all basic readings work without changes.
Advanced BME688 AI/IAQ features (Bosch BSEC) are not supported by this library.
I2C address: `0x77` (secondary).

**SPS30**: Uses the Sensirion I2C driver (`sensirion_i2c_sps30`). All 10 standard
outputs are available via `read_measurement_values_float()`. The library field
`typical_particle_size` is stored as `typical_size_um` in the schema.
I2C address: `0x69`.

**RPi5 timing note**: The SCD41 requires a 0.5 s sleep after every I2C command
write. This is a Raspberry Pi 5 requirement — the I2C controller has different
clock-stretching tolerance from RPi4. Reducing this delay causes
`OSError: [Errno 121] Remote I/O error`.

---

## Initialise DuckDB (run once)

```bash
cd air-monitor
python scripts/init_db.py
```

This creates `data/air_monitor.duckdb` and applies the schema from
`storage/schema.sql`. Safe to re-run — all tables use `CREATE TABLE IF NOT EXISTS`.

---

## Run the collector

```bash
cd air-monitor
python scripts/run_pipeline.py
```

Custom config path:

```bash
python scripts/run_pipeline.py --config /path/to/my_settings.yaml
```

**Expected startup sequence:**
```
SCD41 init:  ~7.5 s  (stop → 1 s → start → 6 s wait)
BME688 init: ~0.5 s
SPS30 init:  ~10.2 s (stop → stabilise)
Total:       ~18 s cold start — normal and expected
```

Stop with `Ctrl+C` — sensors are stopped and the DuckDB connection is
closed cleanly.

---

## MICROBATCH_SIZE

Defined in `config/constants.py`:

```python
MICROBATCH_SIZE: int = 10
```

Every observation belongs to exactly one micro-batch. Once a batch
accumulates `MICROBATCH_SIZE` observations it is closed and a new one begins.
Each observation stores `batch_id` and `row_in_batch` (0-based index).

To override at runtime without editing the constant, set in `settings.yaml`:

```yaml
microbatch_size_override: 20
```

The constant default is always printed at startup for clarity.

---

## Schema overview

### `raw_observations`

One row per sensor collection cycle.

| Column | Type | Description |
|--------|------|-------------|
| ts | TIMESTAMPTZ | UTC timestamp |
| batch_id | VARCHAR | UUID of parent micro-batch |
| row_in_batch | INTEGER | 0-based index within the batch |
| co2_ppm | DOUBLE | CO2 concentration [ppm] |
| scd_temp_c | DOUBLE | SCD41 temperature [°C] |
| scd_humidity_pct | DOUBLE | SCD41 relative humidity [%] |
| bme_temp_c | DOUBLE | BME688 temperature [°C] |
| bme_humidity_pct | DOUBLE | BME688 relative humidity [%] |
| pressure_hpa | DOUBLE | barometric pressure [hPa] |
| gas_resistance_ohm | DOUBLE | BME688 gas resistance [Ω] |
| mass_pm1_0 … mass_pm10 | DOUBLE | PM mass concentrations [µg/m³] |
| number_pm0_5 … number_pm10 | DOUBLE | PM number concentrations [#/cm³] |
| typical_size_um | DOUBLE | typical particle size [µm] |
| sensor_status | VARCHAR | `ok` / `partial` / `failed` |
| scd41_status | VARCHAR | `ok` / `failed` / `disabled` |
| bme688_status | VARCHAR | `ok` / `failed` / `disabled` |
| sps30_status | VARCHAR | `ok` / `failed` / `disabled` / `data_not_ready` |

Failed sensors write NULL for their fields. The pipeline continues.

### `microbatches`

One row per micro-batch. Tracks the batch lifecycle.

| Column | Type | Description |
|--------|------|-------------|
| batch_id | VARCHAR (PK) | UUID |
| start_ts | TIMESTAMPTZ | timestamp of first observation |
| end_ts | TIMESTAMPTZ | timestamp of latest observation |
| record_count | INTEGER | observations written so far |
| is_closed | BOOLEAN | TRUE when record_count = MICROBATCH_SIZE |
| created_at | TIMESTAMPTZ | when the batch row was inserted |
| closed_at | TIMESTAMPTZ | when is_closed became TRUE |

### `runtime_metrics`

One row per collection cycle. Stores all technical metrics in a single table.

Includes:
- per-sensor and pipeline-step timings (ms)
- system resource snapshot (CPU, RAM, disk)
- pipeline outcome (`success`, `error_message`)
- optional power fields (NULL when no power meter is configured)

---

## Batch recovery after restart

When the pipeline starts, `BatchManager.__init__()` queries DuckDB:

```sql
SELECT batch_id, start_ts, record_count
FROM microbatches
WHERE is_closed = FALSE
ORDER BY created_at DESC
LIMIT 1
```

- **Open batch found** → resumes it. The next observation continues at `row_in_batch = record_count`, no rows are duplicated, and the batch fills to `MICROBATCH_SIZE` before closing.
- **No open batch** → the first observation opens a new batch.

This means a crash mid-batch causes no data loss: existing rows are preserved
and the batch is continued from where it stopped.

---

## Limitations of this version

- Single-threaded, synchronous sensor reads (no parallelism)
- No feature engineering or derived metrics
- No rule-based labels
- No machine learning inference
- No cloud sync
- Power measurement fields are always NULL (hardware interface prepared but not wired)
- Gas resistance from BME688 requires heater warm-up; first few values may be 0 Ω

---

## How to extend later

### Add `batch_features` table
After each batch closes in `BatchManager._close_current_batch()`, run aggregate
SQL over `raw_observations WHERE batch_id = ?` (mean, min, max, std per column).
Insert results into a new `batch_features` table. Add `pipeline/feature_engineering.py`.

### Add rule-based labels
Add `pipeline/labeler.py`. Load `config/thresholds.yaml`. After each batch's
features are computed, apply threshold rules and store a label
(`"normal"` / `"warning"` / `"critical"`) in `batch_features.label`.

### Add ML inference
Train a model offline on `batch_features` data (scikit-learn or TFLite).
Add `pipeline/inference.py` with `load_model()` and `predict(features_dict) -> str`.
Call it from the batch-close hook and store `ml_label` in `batch_features`.

### Add batch-level energy analysis
`runtime_metrics` already includes `power_w` and `cycle_total_ms`. Once an
INA219 or USB power meter is wired, query:

```sql
SELECT AVG(power_w), SUM(power_w * cycle_total_ms / 3600000)
FROM runtime_metrics WHERE batch_id = ?
```

Store `avg_power_w` and `energy_wh` in `batch_features`.

### Add cloud sync
Add `pipeline/sync.py`. After each batch closes, serialize `batch_features` as
JSON and push to an MQTT broker or REST endpoint. Delta Lake / Spark can
ingest from there in a separate process.

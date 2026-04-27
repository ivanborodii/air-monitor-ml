-- storage/schema.sql
-- DuckDB schema for the air-monitor pipeline.
-- All statements are idempotent (CREATE TABLE IF NOT EXISTS).
-- Run once via: python scripts/init_db.py
--
-- Three timestamps in raw_observations:
--   ts          - cycle start time (set at beginning of run_cycle())
--   measured_at - exact time sensor reads began (set in sensors/sync.py)
--   loaded_at   - time the whole batch was flushed from JSON buffer to DuckDB
--                 (same for all rows in a batch; shows measurement-to-storage lag)

-- ---------------------------------------------------------------------------
-- Table 1: raw_observations
-- One row per sensor collection cycle.
-- Observations are buffered in data/batch_buffer.json and loaded here in bulk
-- when the batch reaches MICROBATCH_SIZE.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_observations (
    ts                  TIMESTAMPTZ NOT NULL,       -- cycle start time (UTC)
    measured_at         TIMESTAMPTZ NOT NULL,       -- exact time sensor reads began (UTC)
    loaded_at           TIMESTAMPTZ NOT NULL,       -- time this batch was written to DuckDB (UTC)
    batch_id            VARCHAR     NOT NULL,        -- UUID of the parent micro-batch
    row_in_batch        INTEGER     NOT NULL,        -- 0-based index within the batch

    -- SCD41 fields
    co2_ppm             DOUBLE,                     -- CO2 concentration [ppm]
    scd_temp_c          DOUBLE,                     -- temperature from SCD41 [°C]
    scd_humidity_pct    DOUBLE,                     -- relative humidity from SCD41 [%]

    -- BME688 fields
    bme_temp_c          DOUBLE,                     -- temperature from BME688 [°C]
    bme_humidity_pct    DOUBLE,                     -- relative humidity from BME688 [%]
    pressure_hpa        DOUBLE,                     -- barometric pressure [hPa]
    gas_resistance_ohm  DOUBLE,                     -- gas resistance [Ω]

    -- SPS30 fields — all 10 standard outputs
    -- Library: sensirion_i2c_sps30 v1.0.0, read_measurement_values_float()
    mass_pm1_0          DOUBLE,                     -- mass concentration PM1.0 [µg/m³]
    mass_pm2_5          DOUBLE,                     -- mass concentration PM2.5 [µg/m³]
    mass_pm4_0          DOUBLE,                     -- mass concentration PM4.0 [µg/m³]
    mass_pm10           DOUBLE,                     -- mass concentration PM10.0 [µg/m³]
    number_pm0_5        DOUBLE,                     -- number concentration PM0.5 [#/cm³]
    number_pm1_0        DOUBLE,                     -- number concentration PM1.0 [#/cm³]
    number_pm2_5        DOUBLE,                     -- number concentration PM2.5 [#/cm³]
    number_pm4_0        DOUBLE,                     -- number concentration PM4.0 [#/cm³]
    number_pm10         DOUBLE,                     -- number concentration PM10.0 [#/cm³]
    typical_size_um     DOUBLE,                     -- typical particle size [µm] (library: typical_particle_size)

    -- Sensor health flags for this cycle
    sensor_status       VARCHAR,    -- 'ok' | 'partial' | 'failed'
    scd41_status        VARCHAR,    -- 'ok' | 'failed' | 'disabled'
    bme688_status       VARCHAR,    -- 'ok' | 'failed' | 'disabled'
    sps30_status        VARCHAR     -- 'ok' | 'failed' | 'disabled' | 'data_not_ready'
);

-- ---------------------------------------------------------------------------
-- Table 2: microbatches
-- One row per micro-batch, inserted when the batch is flushed from the JSON
-- buffer to DuckDB. Batches are always complete (is_closed=TRUE) when recorded.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS microbatches (
    batch_id        VARCHAR     PRIMARY KEY,         -- UUID string
    start_ts        TIMESTAMPTZ NOT NULL,            -- measured_at of first observation
    end_ts          TIMESTAMPTZ NOT NULL,            -- measured_at of last observation
    record_count    INTEGER     NOT NULL,            -- always equals MICROBATCH_SIZE when inserted
    is_closed       BOOLEAN     NOT NULL DEFAULT TRUE, -- always TRUE (batch is complete on insert)
    created_at      TIMESTAMPTZ NOT NULL,            -- time of DuckDB flush (= loaded_at)
    closed_at       TIMESTAMPTZ NOT NULL             -- same as created_at
);

-- ---------------------------------------------------------------------------
-- Table 3: runtime_metrics
-- One row per collection cycle. Written directly to DuckDB each cycle
-- (not buffered in JSON). Stores timings, system resources, and power fields.
-- Power fields are 0.0 / 'not_configured' when no power meter is connected.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS runtime_metrics (
    ts                      TIMESTAMPTZ NOT NULL,   -- UTC timestamp of the cycle
    cycle_id                VARCHAR     NOT NULL,    -- UUID identifying this specific cycle
    batch_id                VARCHAR,                 -- batch_id assigned during this cycle
    row_in_batch            INTEGER,                 -- row_in_batch assigned during this cycle

    -- Per-sensor and pipeline step timings (milliseconds)
    cycle_total_ms          DOUBLE,                 -- total wall-clock time for the full cycle
    scd41_read_ms           DOUBLE,                 -- time spent in SCD41Sensor.read()
    bme688_read_ms          DOUBLE,                 -- time spent in BME688Sensor.read()
    sps30_read_ms           DOUBLE,                 -- time spent in SPS30Sensor.read()
    db_write_ms             DOUBLE,                 -- time for batch_manager.add() (JSON write + optional DB flush)
    batch_state_update_ms   DOUBLE,                 -- reserved (NULL in current version)
    total_sensor_read_ms    DOUBLE,                 -- sum of all sensor read times

    -- System resource snapshot (psutil)
    cpu_pct                 DOUBLE,                 -- system CPU usage [%]
    ram_used_mb             DOUBLE,                 -- system RAM used [MB]
    ram_percent             DOUBLE,                 -- system RAM usage [%]
    disk_used_gb            DOUBLE,                 -- root disk used [GB]
    disk_free_gb            DOUBLE,                 -- root disk free [GB]
    process_memory_mb       DOUBLE,                 -- this process RSS [MB]

    -- Pipeline outcome
    success                 BOOLEAN,                -- TRUE if observation was stored to JSON buffer successfully
    error_message           VARCHAR,                -- error text if success=FALSE, else NULL

    -- Power measurement fields.
    -- When no power meter is configured: voltage_v/current_a/power_w/cumulative_energy_wh = 0.0,
    -- power_source = 'not_configured'.
    voltage_v               DOUBLE,
    current_a               DOUBLE,
    power_w                 DOUBLE,
    cumulative_energy_wh    DOUBLE,
    power_source            VARCHAR
);

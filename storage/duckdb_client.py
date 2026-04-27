# storage/duckdb_client.py
# DuckDB access layer for the air-monitor pipeline.
# A single persistent connection is opened at construction and reused for
# the entire process lifetime — do not open/close per-insert.

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import duckdb

logger = logging.getLogger(__name__)


class DuckDBClient:
    """Manages the DuckDB connection and all SQL operations."""

    def __init__(self, db_path: str) -> None:
        """
        Open a persistent DuckDB connection to the given file path.
        The schema must already exist (run scripts/init_db.py first).
        """
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = duckdb.connect(db_path)
        logger.info("DuckDB connected: %s", db_path)

    # ------------------------------------------------------------------
    # Transactions
    # ------------------------------------------------------------------

    def begin(self) -> None:
        self._conn.begin()

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    # ------------------------------------------------------------------
    # raw_observations
    # ------------------------------------------------------------------

    def insert_observation(self, obs: dict) -> None:
        """
        Insert one row into raw_observations.
        Expects obs to contain: ts, measured_at, loaded_at, batch_id,
        row_in_batch, all sensor fields, and status fields.
        None values become SQL NULL.
        """
        self._conn.execute(
            """
            INSERT INTO raw_observations (
                ts, measured_at, loaded_at, batch_id, row_in_batch,
                co2_ppm, scd_temp_c, scd_humidity_pct,
                bme_temp_c, bme_humidity_pct, pressure_hpa, gas_resistance_ohm,
                mass_pm1_0, mass_pm2_5, mass_pm4_0, mass_pm10,
                number_pm0_5, number_pm1_0, number_pm2_5, number_pm4_0, number_pm10,
                typical_size_um,
                sensor_status, scd41_status, bme688_status, sps30_status
            ) VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?,
                ?, ?, ?, ?
            )
            """,
            [
                obs["ts"], obs["measured_at"], obs["loaded_at"],
                obs["batch_id"], obs["row_in_batch"],
                obs.get("co2_ppm"), obs.get("scd_temp_c"), obs.get("scd_humidity_pct"),
                obs.get("bme_temp_c"), obs.get("bme_humidity_pct"),
                obs.get("pressure_hpa"), obs.get("gas_resistance_ohm"),
                obs.get("mass_pm1_0"), obs.get("mass_pm2_5"),
                obs.get("mass_pm4_0"), obs.get("mass_pm10"),
                obs.get("number_pm0_5"), obs.get("number_pm1_0"),
                obs.get("number_pm2_5"), obs.get("number_pm4_0"), obs.get("number_pm10"),
                obs.get("typical_size_um"),
                obs.get("sensor_status"), obs.get("scd41_status"),
                obs.get("bme688_status"), obs.get("sps30_status"),
            ],
        )

    # ------------------------------------------------------------------
    # microbatches
    # ------------------------------------------------------------------

    def batch_exists(self, batch_id: str) -> bool:
        """
        Return True if a batch with this ID is already recorded in microbatches.
        Used to skip duplicate flushes after a crash between DB commit and JSON truncation.
        """
        row = self._conn.execute(
            "SELECT 1 FROM microbatches WHERE batch_id = ? LIMIT 1",
            [batch_id],
        ).fetchone()
        return row is not None

    def insert_batch_record(
        self,
        batch_id: str,
        start_ts: datetime,
        end_ts: datetime,
        record_count: int,
        loaded_at: datetime,
    ) -> None:
        """
        Insert a complete batch record. Batches are always closed (is_closed=TRUE)
        when recorded — they are only written to DuckDB when fully flushed from the
        JSON buffer.
        """
        self._conn.execute(
            """
            INSERT INTO microbatches
                (batch_id, start_ts, end_ts, record_count, is_closed, created_at, closed_at)
            VALUES (?, ?, ?, ?, TRUE, ?, ?)
            """,
            [batch_id, start_ts, end_ts, record_count, loaded_at, loaded_at],
        )

    # ------------------------------------------------------------------
    # runtime_metrics
    # ------------------------------------------------------------------

    def insert_runtime_metrics(self, metrics: dict) -> None:
        """Insert one row into runtime_metrics. None values become SQL NULL."""
        self._conn.execute(
            """
            INSERT INTO runtime_metrics (
                ts, cycle_id, batch_id, row_in_batch,
                cycle_total_ms, scd41_read_ms, bme688_read_ms, sps30_read_ms,
                db_write_ms, batch_state_update_ms, total_sensor_read_ms,
                cpu_pct, ram_used_mb, ram_percent,
                disk_used_gb, disk_free_gb, process_memory_mb,
                success, error_message,
                voltage_v, current_a, power_w, cumulative_energy_wh, power_source
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?, ?, ?, ?
            )
            """,
            [
                metrics["ts"], metrics["cycle_id"],
                metrics.get("batch_id"), metrics.get("row_in_batch"),
                metrics.get("cycle_total_ms"), metrics.get("scd41_read_ms"),
                metrics.get("bme688_read_ms"), metrics.get("sps30_read_ms"),
                metrics.get("db_write_ms"), metrics.get("batch_state_update_ms"),
                metrics.get("total_sensor_read_ms"),
                metrics.get("cpu_pct"), metrics.get("ram_used_mb"),
                metrics.get("ram_percent"), metrics.get("disk_used_gb"),
                metrics.get("disk_free_gb"), metrics.get("process_memory_mb"),
                metrics.get("success"), metrics.get("error_message"),
                metrics.get("voltage_v"), metrics.get("current_a"),
                metrics.get("power_w"), metrics.get("cumulative_energy_wh"),
                metrics.get("power_source"),
            ],
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()
        logger.info("DuckDB connection closed")

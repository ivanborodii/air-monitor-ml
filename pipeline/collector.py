# pipeline/collector.py
# Orchestrates one complete data collection cycle.
#
# Cycle flow:
#   1. Read all sensors (fault-isolated, individually timed)
#   2. Build the full observation dict (ts, measured_at, sensor fields)
#   3. Pass observation to BatchManager.add() — writes to JSON buffer,
#      flushes to DuckDB when batch reaches MICROBATCH_SIZE
#   4. Collect system and power metrics
#   5. Write one runtime_metrics row directly to DuckDB (not buffered)
#
# The JSON buffer (not DuckDB) is the write target per cycle.
# DuckDB raw_observations is only written at batch-flush time.
# runtime_metrics is written every cycle regardless.

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import psutil

from pipeline.batching import BatchManager
from pipeline.metrics import collect_system_metrics
from pipeline.power import PowerMeter
from sensors import sync
from storage.duckdb_client import DuckDBClient

logger = logging.getLogger(__name__)


class PipelineCollector:
    """Executes one full collection cycle per call to run_cycle()."""

    def __init__(
        self,
        sensors: dict,
        db_client: DuckDBClient,
        batch_manager: BatchManager,
        power_meter: PowerMeter,
        config: dict,
        process: psutil.Process,
    ) -> None:
        self._sensors = sensors
        self._enabled = config.get("enabled_sensors", {})
        self._db = db_client
        self._batches = batch_manager
        self._power = power_meter
        self._enable_runtime_metrics = config.get("enable_runtime_metrics", True)
        self._process = process

    def run_cycle(self) -> dict:
        """
        Execute one full pipeline cycle.

        Returns a summary dict for logging by the caller.
        """
        cycle_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc)
        cycle_start = time.monotonic()

        success = False
        error_message: Optional[str] = None
        batch_id: Optional[str] = None
        row_in_batch: Optional[int] = None
        db_write_ms: Optional[float] = None

        # ------------------------------------------------------------------
        # 1. Read all sensors (fault-isolated)
        #    obs_fields includes measured_at (exact sensor-read timestamp)
        # ------------------------------------------------------------------
        obs_fields, timings = sync.read_all_sensors(self._sensors, self._enabled)

        # ------------------------------------------------------------------
        # 2. Build full observation dict
        #    batch_id and row_in_batch are assigned by BatchManager.add()
        # ------------------------------------------------------------------
        observation = {
            "ts": ts,
            **obs_fields,           # includes measured_at and all sensor fields
        }

        # ------------------------------------------------------------------
        # 3. Add to JSON buffer (flushes to DuckDB when batch is full)
        # ------------------------------------------------------------------
        try:
            t_write = time.monotonic()
            batch_id, row_in_batch, batch_flushed = self._batches.add(observation)
            db_write_ms = (time.monotonic() - t_write) * 1000
            success = True

            if batch_flushed:
                logger.info(
                    "Cycle OK | batch=%s row=%d sensor=%s | buffer+flush=%.1fms [BATCH FLUSHED]",
                    batch_id[:8],
                    row_in_batch,
                    obs_fields.get("sensor_status", "?"),
                    db_write_ms,
                )
            else:
                logger.info(
                    "Cycle OK | batch=%s row=%d sensor=%s | buffer=%.1fms [%d/%d]",
                    batch_id[:8],
                    row_in_batch,
                    obs_fields.get("sensor_status", "?"),
                    db_write_ms,
                    self._batches.current_buffer_size,
                    self._batches._size,
                )

        except Exception as exc:
            error_message = str(exc)
            # Recover batch_id from the manager even on failure for metrics logging
            batch_id = self._batches.current_batch_id
            row_in_batch = self._batches.current_buffer_size
            logger.error(
                "Cycle FAILED (buffer/flush error) | batch=%s | error: %s",
                batch_id[:8] if batch_id else "?",
                exc,
            )

        # ------------------------------------------------------------------
        # 4. Collect system and power metrics
        # ------------------------------------------------------------------
        sys_metrics = collect_system_metrics(self._process)
        power_reading = self._power.read()

        # ------------------------------------------------------------------
        # 5. Compute cycle total time
        # ------------------------------------------------------------------
        cycle_total_ms = (time.monotonic() - cycle_start) * 1000

        # ------------------------------------------------------------------
        # 6. Write runtime_metrics directly to DuckDB (not buffered)
        # ------------------------------------------------------------------
        if self._enable_runtime_metrics:
            runtime_metrics = {
                "ts": ts,
                "cycle_id": cycle_id,
                "batch_id": batch_id,
                "row_in_batch": row_in_batch,
                "cycle_total_ms": cycle_total_ms,
                "scd41_read_ms": timings.get("scd41_read_ms"),
                "bme688_read_ms": timings.get("bme688_read_ms"),
                "sps30_read_ms": timings.get("sps30_read_ms"),
                "db_write_ms": db_write_ms,
                "batch_state_update_ms": None,   # folded into db_write_ms in this version
                "total_sensor_read_ms": timings.get("total_sensor_read_ms"),
                **sys_metrics,
                "success": success,
                "error_message": error_message,
                **power_reading,
            }
            try:
                self._db.insert_runtime_metrics(runtime_metrics)
            except Exception as exc:
                logger.warning("Failed to write runtime_metrics: %s", exc)

        return {
            "cycle_id": cycle_id,
            "ts": ts,
            "batch_id": batch_id,
            "row_in_batch": row_in_batch,
            "success": success,
            "cycle_total_ms": cycle_total_ms,
            "sensor_status": obs_fields.get("sensor_status"),
        }

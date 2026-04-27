#!/usr/bin/env python3
# scripts/run_pipeline.py
# Main entry point for the air-monitor pipeline.
#
# Usage:
#   cd air-monitor
#   python scripts/run_pipeline.py [--config config/settings.yaml]
#
# Prerequisites:
#   python scripts/init_db.py   # run once to create the DuckDB schema
#
# Startup time:
#   SCD41 init: ~7.5 s (stop + 1 s + start + 6 s)
#   SPS30 init: ~10.2 s (stop + stabilise)
#   Total cold start: ~18 s — this is expected and normal.
#
# Graceful shutdown:
#   Ctrl+C (SIGINT) or SIGTERM → sensors stopped, DB closed, exit 0.

import argparse
import logging
import logging.handlers
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psutil
import yaml

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.constants import MICROBATCH_SIZE
from pipeline.batching import BatchManager
from pipeline.collector import PipelineCollector
from pipeline.power import build_power_meter
from sensors.bme688 import BME688Sensor
from sensors.scd41 import SCD41Sensor
from sensors.sps30 import SPS30Sensor
from storage.duckdb_client import DuckDBClient


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_path: str) -> None:
    """Configure root logger with console output and a rotating file handler."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setFormatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load and return settings.yaml as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Sensor initialisation
# ---------------------------------------------------------------------------

def initialize_sensors(config: dict) -> dict:
    """
    Instantiate each enabled sensor. Failures are caught individually so one
    broken sensor does not prevent the others from starting.

    Returns a dict {"scd41": instance|None, "bme688": instance|None, "sps30": instance|None}.
    """
    enabled = config.get("enabled_sensors", {})
    sensors: dict = {"scd41": None, "bme688": None, "sps30": None}
    logger = logging.getLogger(__name__)

    if enabled.get("scd41", True):
        try:
            logger.info("Initialising SCD41 sensor...")
            sensors["scd41"] = SCD41Sensor()
            logger.info("SCD41 initialised successfully")
        except Exception as exc:
            logger.error("SCD41 initialisation failed: %s — sensor disabled for this run", exc)
    else:
        logger.info("SCD41 disabled in config")

    if enabled.get("bme688", True):
        try:
            logger.info("Initialising BME688 sensor...")
            sensors["bme688"] = BME688Sensor()
            logger.info("BME688 initialised successfully")
        except Exception as exc:
            logger.error("BME688 initialisation failed: %s — sensor disabled for this run", exc)
    else:
        logger.info("BME688 disabled in config")

    if enabled.get("sps30", True):
        try:
            logger.info("Initialising SPS30 sensor...")
            sensors["sps30"] = SPS30Sensor()
            logger.info("SPS30 initialised successfully")
        except Exception as exc:
            logger.error("SPS30 initialisation failed: %s — sensor disabled for this run", exc)
    else:
        logger.info("SPS30 disabled in config")

    active = [k for k, v in sensors.items() if v is not None]
    logger.info("Active sensors: %s", active if active else "NONE (pipeline will write NULL values)")

    return sensors


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def shutdown(sensors: dict, db_client: Optional[DuckDBClient]) -> None:
    """Stop sensors and close the database connection cleanly."""
    logger = logging.getLogger(__name__)
    logger.info("Shutting down pipeline...")

    for name, sensor in sensors.items():
        if sensor is not None:
            try:
                sensor.close()
            except Exception as exc:
                logger.warning("%s close error: %s", name.upper(), exc)

    if db_client is not None:
        try:
            db_client.close()
        except Exception as exc:
            logger.warning("DuckDB close error: %s", exc)

    logger.info("Pipeline stopped cleanly")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Air quality monitoring pipeline")
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    setup_logging(config.get("log_path", "data/logs/pipeline.log"))
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Air-monitor pipeline starting")
    logger.info("Config: %s", args.config)

    # Resolve micro-batch size: YAML override takes precedence over the constant
    microbatch_size = config.get("microbatch_size_override") or MICROBATCH_SIZE
    logger.info("MICROBATCH_SIZE = %d (constant default: %d)", microbatch_size, MICROBATCH_SIZE)

    collection_interval = config.get("collection_interval_seconds", 30)
    logger.info("Collection interval: %d s", collection_interval)

    # SIGTERM handler for systemd / process managers
    signal.signal(signal.SIGTERM, lambda sig, frame: sys.exit(0))

    sensors: dict = {}
    db_client: Optional[DuckDBClient] = None

    try:
        # Initialise sensors (this takes ~18 s on a cold start)
        sensors = initialize_sensors(config)

        # Open DuckDB (schema must already exist — run init_db.py first)
        db_path = config.get("db_path", "data/air_monitor.duckdb")
        logger.info("Opening DuckDB: %s", db_path)
        db_client = DuckDBClient(db_path)

        # Batch manager — buffers observations in JSON, flushes to DuckDB on batch fill.
        # Crash recovery: reads existing JSON buffer on startup.
        buffer_path = config.get("json_buffer_path", "data/batch_buffer.json")
        batch_manager = BatchManager(db_client, microbatch_size, buffer_path)

        # Power meter
        power_meter = build_power_meter(config)

        # psutil Process handle (created once, reused each cycle)
        process = psutil.Process(os.getpid())
        # Prime the CPU percentage counter (first call always returns 0.0)
        psutil.cpu_percent(interval=None)

        collector = PipelineCollector(
            sensors=sensors,
            db_client=db_client,
            batch_manager=batch_manager,
            power_meter=power_meter,
            config=config,
            process=process,
        )

        logger.info("Pipeline running - press Ctrl+C to stop")
        logger.info("=" * 60)

        while True:
            t_cycle_wall = time.monotonic()

            try:
                result = collector.run_cycle()
            except Exception as exc:
                # Unexpected top-level exception — log and continue
                logger.error("Unhandled cycle exception: %s", exc, exc_info=True)
                result = {"cycle_total_ms": None, "success": False}

            elapsed = time.monotonic() - t_cycle_wall
            sleep_s = max(0.0, collection_interval - elapsed)
            logger.debug("Cycle done in %.1f s, sleeping %.1f s", elapsed, sleep_s)
            time.sleep(sleep_s)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except SystemExit:
        logger.info("SIGTERM received")
    except Exception as exc:
        logger.critical("Fatal error: %s", exc, exc_info=True)
        sys.exit(1)
    finally:
        shutdown(sensors, db_client)


if __name__ == "__main__":
    main()

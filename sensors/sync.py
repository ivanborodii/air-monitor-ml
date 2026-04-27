# sensors/sync.py
# Coordinated, fault-isolated read of all three sensors in a single cycle.
# Each sensor is read sequentially and individually timed. A failure in one
# sensor does not affect the others — its fields become None and its status
# is recorded. The rest of the pipeline receives a single merged dict.

import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sensors.bme688 import BME688Sensor
    from sensors.scd41 import SCD41Sensor
    from sensors.sps30 import SPS30Sensor

logger = logging.getLogger(__name__)


def read_all_sensors(
    sensors: dict,
    enabled: dict,
) -> tuple[dict, dict]:
    """
    Read all enabled sensors and return merged observation fields + per-sensor timings.

    Parameters
    ----------
    sensors : dict
        Keys: "scd41", "bme688", "sps30". Values: sensor instances or None.
    enabled : dict
        Keys: "scd41", "bme688", "sps30". Values: bool.

    Returns
    -------
    obs_fields : dict
        All sensor value fields plus scd41_status, bme688_status, sps30_status,
        sensor_status, and measured_at (exact UTC timestamp of measurement start).
        Does NOT include ts, batch_id, or row_in_batch.
    timings : dict
        scd41_read_ms, bme688_read_ms, sps30_read_ms, total_sensor_read_ms.
    """
    obs_fields: dict = {}
    timings: dict = {
        "scd41_read_ms": None,
        "bme688_read_ms": None,
        "sps30_read_ms": None,
        "total_sensor_read_ms": None,
    }

    # Capture the exact moment sensor reads begin.
    # This is stored as measured_at in raw_observations to distinguish the
    # actual measurement time from the cycle start time (ts) and the DuckDB
    # load time (loaded_at).
    measured_at = datetime.now(timezone.utc)
    t_total_start = time.monotonic()

    # ------------------------------------------------------------------
    # SCD41
    # ------------------------------------------------------------------
    scd41_status: str
    if not enabled.get("scd41", True) or sensors.get("scd41") is None:
        obs_fields.update({"co2_ppm": None, "scd_temp_c": None, "scd_humidity_pct": None})
        scd41_status = "disabled" if not enabled.get("scd41", True) else "unavailable"
    else:
        t0 = time.monotonic()
        try:
            data = sensors["scd41"].read()
            timings["scd41_read_ms"] = (time.monotonic() - t0) * 1000
            obs_fields.update(data)
            if all(v is not None for v in data.values()):
                scd41_status = "ok"
            else:
                scd41_status = "failed"
                logger.warning("SCD41: read returned partial None values")
        except Exception as exc:
            timings["scd41_read_ms"] = (time.monotonic() - t0) * 1000
            logger.warning("SCD41: unexpected exception in read(): %s", exc)
            obs_fields.update({"co2_ppm": None, "scd_temp_c": None, "scd_humidity_pct": None})
            scd41_status = "failed"

    # ------------------------------------------------------------------
    # BME688
    # ------------------------------------------------------------------
    bme688_status: str
    if not enabled.get("bme688", True) or sensors.get("bme688") is None:
        obs_fields.update({
            "bme_temp_c": None, "bme_humidity_pct": None,
            "pressure_hpa": None, "gas_resistance_ohm": None,
        })
        bme688_status = "disabled" if not enabled.get("bme688", True) else "unavailable"
    else:
        t0 = time.monotonic()
        try:
            data = sensors["bme688"].read()
            timings["bme688_read_ms"] = (time.monotonic() - t0) * 1000
            obs_fields.update(data)
            if all(v is not None for v in data.values()):
                bme688_status = "ok"
            else:
                bme688_status = "failed"
                logger.warning("BME688: read returned partial None values")
        except Exception as exc:
            timings["bme688_read_ms"] = (time.monotonic() - t0) * 1000
            logger.warning("BME688: unexpected exception in read(): %s", exc)
            obs_fields.update({
                "bme_temp_c": None, "bme_humidity_pct": None,
                "pressure_hpa": None, "gas_resistance_ohm": None,
            })
            bme688_status = "failed"

    # ------------------------------------------------------------------
    # SPS30
    # ------------------------------------------------------------------
    sps30_status: str
    _sps30_null = {
        "mass_pm1_0": None, "mass_pm2_5": None, "mass_pm4_0": None, "mass_pm10": None,
        "number_pm0_5": None, "number_pm1_0": None, "number_pm2_5": None,
        "number_pm4_0": None, "number_pm10": None, "typical_size_um": None,
    }
    if not enabled.get("sps30", True) or sensors.get("sps30") is None:
        obs_fields.update(_sps30_null)
        sps30_status = "disabled" if not enabled.get("sps30", True) else "unavailable"
    else:
        t0 = time.monotonic()
        try:
            data = sensors["sps30"].read()
            timings["sps30_read_ms"] = (time.monotonic() - t0) * 1000
            # SPS30 may return _data_not_ready flag (not a hardware failure)
            if data.pop("_data_not_ready", False):
                obs_fields.update(_sps30_null)
                sps30_status = "data_not_ready"
                logger.debug("SPS30: data not ready this cycle")
            elif all(v is not None for v in data.values()):
                obs_fields.update(data)
                sps30_status = "ok"
            else:
                obs_fields.update(data)
                sps30_status = "failed"
                logger.warning("SPS30: read returned partial None values")
        except Exception as exc:
            timings["sps30_read_ms"] = (time.monotonic() - t0) * 1000
            logger.warning("SPS30: unexpected exception in read(): %s", exc)
            obs_fields.update(_sps30_null)
            sps30_status = "failed"

    # ------------------------------------------------------------------
    # Aggregate sensor_status
    # ------------------------------------------------------------------
    all_statuses = [scd41_status, bme688_status, sps30_status]
    active = [s for s in all_statuses if s not in ("disabled", "unavailable")]
    if not active:
        sensor_status = "failed"
    elif all(s == "ok" for s in active):
        sensor_status = "ok"
    elif all(s in ("failed", "data_not_ready") for s in active):
        sensor_status = "failed"
    else:
        sensor_status = "partial"

    obs_fields["scd41_status"] = scd41_status
    obs_fields["bme688_status"] = bme688_status
    obs_fields["sps30_status"] = sps30_status
    obs_fields["sensor_status"] = sensor_status
    obs_fields["measured_at"] = measured_at

    # ------------------------------------------------------------------
    # Total sensor read time
    # ------------------------------------------------------------------
    timings["total_sensor_read_ms"] = (time.monotonic() - t_total_start) * 1000

    if sensor_status != "ok":
        logger.warning(
            "Sensor read completed with status=%s (scd41=%s, bme688=%s, sps30=%s)",
            sensor_status, scd41_status, bme688_status, sps30_status,
        )

    return obs_fields, timings

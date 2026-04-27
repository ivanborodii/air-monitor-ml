# pipeline/power.py
# Optional power measurement interface.
#
# When enable_power_metrics=False (the default), NullPowerMeter returns
# zero/placeholder values so no columns in runtime_metrics are ever NULL:
#   voltage_v / current_a / power_w / cumulative_energy_wh = 0.0
#   power_source = "not_configured"
#
# To add a real power meter:
#   1. Subclass PowerMeter and implement read().
#   2. Return actual measurements.
#   3. Register the class in build_power_meter() below.
#
# Example future meters:
#   - INA219 (I2C current/voltage): pip install adafruit-circuitpython-ina219
#   - USB power meter with serial protocol

import logging

logger = logging.getLogger(__name__)


class PowerMeter:
    """
    Base power meter.

    Returns zero/placeholder values for all fields when no physical power
    meter is connected. power_source="not_configured" signals that these
    readings are placeholders, not real measurements.
    """

    def read(self) -> dict:
        """
        Return power measurement fields.

        Returns
        -------
        dict with keys:
            voltage_v            - supply voltage [V]  (0.0 when not configured)
            current_a            - drawn current [A]   (0.0 when not configured)
            power_w              - instantaneous power [W] (0.0 when not configured)
            cumulative_energy_wh - cumulative energy [Wh]  (0.0 when not configured)
            power_source         - 'not_configured' when no meter is connected
        """
        return {
            "voltage_v": 0.0,
            "current_a": 0.0,
            "power_w": 0.0,
            "cumulative_energy_wh": 0.0,
            "power_source": "not_configured",
        }


# ---------------------------------------------------------------------------
# Placeholder for future INA219 implementation
# ---------------------------------------------------------------------------
# class INA219PowerMeter(PowerMeter):
#     """
#     INA219 I2C current/voltage sensor.
#     Requires: pip install adafruit-circuitpython-ina219 adafruit-blinka
#     """
#     def __init__(self) -> None:
#         import board, busio
#         from adafruit_ina219 import INA219
#         i2c = busio.I2C(board.SCL, board.SDA)
#         self._ina = INA219(i2c)
#         self._cumulative_wh: float = 0.0
#         self._last_ts: Optional[float] = None
#
#     def read(self) -> dict:
#         import time
#         voltage = self._ina.bus_voltage + self._ina.shunt_voltage / 1000
#         current = self._ina.current / 1000   # mA -> A
#         power = voltage * current
#         now = time.monotonic()
#         if self._last_ts is not None:
#             self._cumulative_wh += power * (now - self._last_ts) / 3600
#         self._last_ts = now
#         return {
#             "voltage_v": voltage,
#             "current_a": current,
#             "power_w": power,
#             "cumulative_energy_wh": self._cumulative_wh,
#             "power_source": "ina219",
#         }


def build_power_meter(config: dict) -> PowerMeter:
    """
    Factory. Returns the appropriate PowerMeter based on config.

    When enable_power_metrics=False or power_meter_type=null, returns the base
    PowerMeter (zeros + 'not_configured'). Raises NotImplementedError for
    recognised but not-yet-implemented meter types.
    """
    if not config.get("enable_power_metrics", False):
        logger.debug("Power metrics disabled - using null power meter (zeros)")
        return PowerMeter()

    meter_type = config.get("power_meter_type")
    if meter_type is None:
        logger.warning(
            "enable_power_metrics=true but power_meter_type is null - using null meter"
        )
        return PowerMeter()

    if meter_type == "ina219":
        raise NotImplementedError(
            "INA219 power meter support is not yet implemented. "
            "Uncomment INA219PowerMeter in pipeline/power.py and wire it here."
        )

    raise NotImplementedError(f"Unknown power_meter_type: '{meter_type}'")

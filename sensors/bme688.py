# sensors/bme688.py
# Wrapper for the BME688 environmental sensor.
#
# Library note:
#   This module uses the `bme680` Python library (v2.0.0).
#   The BME688 is pin-compatible with the BME680 and the library works
#   for all basic readings (temperature, humidity, pressure, gas resistance)
#   without modification. Advanced BME688 features (BSEC AI / IAQ) require
#   Bosch Sensortec's BSEC library and are not supported here.
#
# I2C wiring: secondary address 0x77 (bme680.I2C_ADDR_SECONDARY), bus /dev/i2c-1
#
# Gas resistance note:
#   The first few readings after power-on may return 0 Ω or very low values
#   while the heater stabilises. This is normal and expected.

import logging
from typing import Optional

import bme680

logger = logging.getLogger(__name__)


class BME688Sensor:
    """
    Adapter for the BME688 sensor using the bme680 v2.0.0 library.

    The library initialises calibration and sensor configuration in __init__,
    so no additional setup step is required.
    """

    def __init__(self) -> None:
        """
        Instantiate the bme680 driver at I2C secondary address (0x77).
        The library runs a soft reset, reads calibration data, and configures
        default oversampling and filter settings internally.
        """
        logger.info("BME688: initialising at I2C_ADDR_SECONDARY (0x77)")
        self._sensor = bme680.BME680(bme680.I2C_ADDR_SECONDARY)
        logger.info("BME688: ready")

    def read(self) -> dict:
        """
        Trigger a forced-mode measurement and return sensor values.

        The library's get_sensor_data() polls internally until new data is
        available (up to ~10 attempts) and returns True on success.
        Data is accessed via self._sensor.data.*.

        Returns a dict with bme_temp_c, bme_humidity_pct, pressure_hpa,
        gas_resistance_ohm.
        All values are None on failure — never raises.
        """
        try:
            if self._sensor.get_sensor_data():
                return {
                    "bme_temp_c": float(self._sensor.data.temperature),
                    "bme_humidity_pct": float(self._sensor.data.humidity),
                    "pressure_hpa": float(self._sensor.data.pressure),
                    "gas_resistance_ohm": float(self._sensor.data.gas_resistance),
                }
            else:
                logger.warning("BME688: get_sensor_data() returned False (no new data)")
                return {
                    "bme_temp_c": None,
                    "bme_humidity_pct": None,
                    "pressure_hpa": None,
                    "gas_resistance_ohm": None,
                }
        except Exception as exc:
            logger.warning("BME688 read error: %s", exc)
            return {
                "bme_temp_c": None,
                "bme_humidity_pct": None,
                "pressure_hpa": None,
                "gas_resistance_ohm": None,
            }

    def healthcheck(self) -> bool:
        """Return True if a read produces non-None values for all fields."""
        result = self.read()
        return all(v is not None for v in result.values())

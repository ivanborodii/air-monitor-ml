# sensors/scd41.py
# Wrapper for the SCD41 CO2 / temperature / humidity sensor.
# Hardware access is isolated here. The rest of the pipeline uses only
# SCD41Sensor.read() and SCD41Sensor.healthcheck().
#
# I2C wiring: address 0x62, bus /dev/i2c-1 (Raspberry Pi 5)
# Library:    smbus2 (raw I2C — no third-party SCD41 driver needed)
#
# RPi5 note: The I2C controller on Raspberry Pi 5 requires a 0.5 s sleep
# after every command write. Shorter delays cause OSError: [Errno 121]
# (Remote I/O error). Do not reduce SCD41_CMD_SLEEP_S.

import logging
import time
from typing import Optional

from smbus2 import SMBus, i2c_msg

from config.constants import (
    SCD41_CMD_READ,
    SCD41_CMD_SLEEP_S,
    SCD41_CMD_START,
    SCD41_CMD_STOP,
    SCD41_I2C_ADDR,
    SCD41_INIT_WAIT_S,
    SCD41_READ_SETTLE_S,
    SCD41_STOP_SLEEP_S,
)

logger = logging.getLogger(__name__)


class SCD41Sensor:
    """
    Adapter for the SCD41 CO2 sensor via raw I2C (smbus2).

    Lifecycle:
      1. __init__() opens the SMBus and runs the init sequence.
      2. read() is called each pipeline cycle.
      3. close() stops measurement and releases the bus.
    """

    def __init__(self, bus_number: int = 1) -> None:
        """
        Open the SMBus and run the SCD41 init sequence:
          stop_periodic_measurement → wait 1 s → start_periodic_measurement → wait 6 s

        The 6-second wait is required before the first valid sample is available.
        Total __init__ time: ~7.5 s.
        """
        logger.info("SCD41: opening I2C bus %d, address 0x%02X", bus_number, SCD41_I2C_ADDR)
        self._bus = SMBus(bus_number)
        self._address = SCD41_I2C_ADDR

        # Stop any ongoing measurement before (re)initialising
        self._send_command(SCD41_CMD_STOP)
        time.sleep(SCD41_STOP_SLEEP_S)

        # Start periodic measurement
        self._send_command(SCD41_CMD_START)

        logger.info("SCD41: waiting %.0f s for first sample...", SCD41_INIT_WAIT_S)
        time.sleep(SCD41_INIT_WAIT_S)
        logger.info("SCD41: ready")

    def _send_command(self, cmd: int) -> None:
        """
        Write a 16-bit command to the sensor.
        Sleeps SCD41_CMD_SLEEP_S (0.5 s) after every write — mandatory on RPi5.
        """
        high = (cmd >> 8) & 0xFF
        low = cmd & 0xFF
        msg = i2c_msg.write(self._address, [high, low])
        self._bus.i2c_rdwr(msg)
        time.sleep(SCD41_CMD_SLEEP_S)

    def read(self) -> dict:
        """
        Trigger a measurement read and return parsed sensor values.

        The SCD41 returns 9 bytes in response to the read_measurement command:
          bytes 0-1: CO2 raw value
          byte  2:   CRC (ignored)
          bytes 3-4: temperature raw value
          byte  5:   CRC (ignored)
          bytes 6-7: humidity raw value
          byte  8:   CRC (ignored)

        Parsing formulas (from Sensirion SCD4x datasheet):
          co2   = raw_co2   (direct ppm value)
          temp  = -45 + 175 * raw_temp / 65536
          humid = 100 * raw_humid / 65536

        Returns a dict with co2_ppm, scd_temp_c, scd_humidity_pct.
        All values are None on any I2C error — never raises.
        """
        try:
            self._send_command(SCD41_CMD_READ)
            time.sleep(SCD41_READ_SETTLE_S)

            msg = i2c_msg.read(self._address, 9)
            self._bus.i2c_rdwr(msg)
            data = list(msg)

            co2 = (data[0] << 8) | data[1]
            temp_raw = (data[3] << 8) | data[4]
            humi_raw = (data[6] << 8) | data[7]

            temperature = -45.0 + 175.0 * temp_raw / 65536.0
            humidity = 100.0 * humi_raw / 65536.0

            return {
                "co2_ppm": float(co2),
                "scd_temp_c": float(temperature),
                "scd_humidity_pct": float(humidity),
            }
        except Exception as exc:
            logger.warning("SCD41 read error: %s", exc)
            return {"co2_ppm": None, "scd_temp_c": None, "scd_humidity_pct": None}

    def healthcheck(self) -> bool:
        """Return True if a read returns non-None values for all three fields."""
        result = self.read()
        return all(v is not None for v in result.values())

    def close(self) -> None:
        """Stop periodic measurement and release the SMBus."""
        try:
            self._send_command(SCD41_CMD_STOP)
        except Exception as exc:
            logger.warning("SCD41 stop error during close: %s", exc)
        try:
            self._bus.close()
        except Exception as exc:
            logger.warning("SCD41 SMBus close error: %s", exc)
        logger.info("SCD41: closed")

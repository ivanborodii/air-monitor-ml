# sensors/sps30.py
# Wrapper for the SPS30 particulate matter sensor.
#
# Library: sensirion_i2c_sps30 v1.0.0
# I2C wiring: address 0x69, port /dev/i2c-1 (Raspberry Pi 5)
#
# IMPORTANT — transceiver lifetime:
#   LinuxI2cTransceiver opens a file descriptor to /dev/i2c-1 in __init__
#   (do_open=True is the default). It must remain open for the entire pipeline
#   lifetime — do NOT use it as a context manager at the method level, as that
#   closes the fd when the method exits and all subsequent reads would raise
#   OSError: [Errno 9] Bad file descriptor.
#
#   Instead: open in __init__, close explicitly in close().
#
# SPS30 all-output note:
#   read_measurement_values_float() returns a 10-tuple:
#     (mc_1p0, mc_2p5, mc_4p0, mc_10p0,
#      nc_0p5, nc_1p0, nc_2p5, nc_4p0, nc_10p0,
#      typical_particle_size)
#   All 10 outputs are available with this library version.
#   The library field "typical_particle_size" is stored as "typical_size_um"
#   in the pipeline schema.

import logging
import time

from sensirion_driver_adapters.i2c_adapter.i2c_channel import I2cChannel
from sensirion_i2c_driver import CrcCalculator, I2cConnection, LinuxI2cTransceiver
from sensirion_i2c_sps30.commands import OutputFormat
from sensirion_i2c_sps30.device import Sps30Device

from config.constants import (
    SPS30_I2C_ADDR,
    SPS30_I2C_PORT,
    SPS30_STARTUP_WAIT_S,
    SPS30_STOP_SETTLE_S,
)

logger = logging.getLogger(__name__)

# Keys returned by read() — matches the raw_observations schema
_EMPTY_RESULT: dict = {
    "mass_pm1_0": None,
    "mass_pm2_5": None,
    "mass_pm4_0": None,
    "mass_pm10": None,
    "number_pm0_5": None,
    "number_pm1_0": None,
    "number_pm2_5": None,
    "number_pm4_0": None,
    "number_pm10": None,
    "typical_size_um": None,
}


class SPS30Sensor:
    """
    Adapter for the SPS30 particulate matter sensor using the Sensirion I2C driver.

    The I2C transceiver file descriptor is opened once at __init__ and kept
    open until close() is called. Do not use a context manager per read cycle.
    """

    def __init__(self) -> None:
        """
        Open the I2C transceiver, build the driver stack, start measurement,
        and wait for the sensor to stabilise (~10 s).

        Total __init__ time: ~10.2 s.
        """
        logger.info(
            "SPS30: opening I2C transceiver on %s, address 0x%02X",
            SPS30_I2C_PORT,
            SPS30_I2C_ADDR,
        )

        # Open the file descriptor immediately (do_open=True is the default).
        # We manage the fd lifetime manually — no context manager.
        self._transceiver = LinuxI2cTransceiver(SPS30_I2C_PORT)

        channel = I2cChannel(
            I2cConnection(self._transceiver),
            slave_address=SPS30_I2C_ADDR,
            crc=CrcCalculator(8, 0x31, 0xFF, 0x00),
        )
        self._sensor = Sps30Device(channel)

        # Stop any in-progress measurement from a previous run
        try:
            self._sensor.stop_measurement()
            time.sleep(SPS30_STOP_SETTLE_S)
        except Exception as exc:
            logger.debug("SPS30: stop_measurement() during init: %s (non-fatal)", exc)

        logger.info("SPS30: starting measurement in float mode")
        self._sensor.start_measurement(OutputFormat.OUTPUT_FORMAT_FLOAT)

        logger.info("SPS30: waiting %.0f s for stabilisation...", SPS30_STARTUP_WAIT_S)
        time.sleep(SPS30_STARTUP_WAIT_S)
        logger.info("SPS30: ready")

    def read(self) -> dict:
        """
        Read the latest particulate matter data from the sensor.

        First checks the data-ready flag (0 = not ready, 1 = new data available).
        If data is not ready, returns all-None with sps30_data_not_ready=True so
        the caller can distinguish a timing issue from a hardware failure.

        Returns a dict with all 10 SPS30 output fields.
        Never raises.
        """
        try:
            data_ready = self._sensor.read_data_ready_flag()
            if not data_ready:
                logger.debug("SPS30: data not ready this cycle")
                result = dict(_EMPTY_RESULT)
                result["_data_not_ready"] = True
                return result

            (
                mc_1p0,
                mc_2p5,
                mc_4p0,
                mc_10p0,
                nc_0p5,
                nc_1p0,
                nc_2p5,
                nc_4p0,
                nc_10p0,
                typical_particle_size,
            ) = self._sensor.read_measurement_values_float()

            return {
                "mass_pm1_0": float(mc_1p0),
                "mass_pm2_5": float(mc_2p5),
                "mass_pm4_0": float(mc_4p0),
                "mass_pm10": float(mc_10p0),
                "number_pm0_5": float(nc_0p5),
                "number_pm1_0": float(nc_1p0),
                "number_pm2_5": float(nc_2p5),
                "number_pm4_0": float(nc_4p0),
                "number_pm10": float(nc_10p0),
                "typical_size_um": float(typical_particle_size),  # library: typical_particle_size
            }
        except Exception as exc:
            logger.warning("SPS30 read error: %s", exc)
            return dict(_EMPTY_RESULT)

    def healthcheck(self) -> bool:
        """Return True if the data-ready flag is set (new data is available)."""
        try:
            return bool(self._sensor.read_data_ready_flag())
        except Exception:
            return False

    def close(self) -> None:
        """Stop measurement and close the I2C transceiver file descriptor."""
        try:
            self._sensor.stop_measurement()
        except Exception as exc:
            logger.warning("SPS30 stop_measurement() error during close: %s", exc)
        try:
            self._transceiver.close()
        except Exception as exc:
            logger.warning("SPS30 transceiver close error: %s", exc)
        logger.info("SPS30: closed")

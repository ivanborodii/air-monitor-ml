# config/constants.py
# Compile-time constants for the air-monitor pipeline.
# Hardware-specific values are documented inline so they can be adjusted
# if the sensor wiring or Raspberry Pi version changes.

# ---------------------------------------------------------------------------
# Micro-batching
# ---------------------------------------------------------------------------

MICROBATCH_SIZE: int = 10
# Number of observations per micro-batch.
# Can be overridden at runtime via settings.yaml:microbatch_size_override,
# but this constant is always the canonical default and must remain visible here.

# ---------------------------------------------------------------------------
# SCD41 — CO2 / temperature / humidity sensor
# I2C address 0x62, connected to /dev/i2c-1 on Raspberry Pi 5
# ---------------------------------------------------------------------------

SCD41_I2C_ADDR: int = 0x62

# 16-bit command words (big-endian, sent as two bytes)
SCD41_CMD_STOP: int = 0x3F86   # stop_periodic_measurement
SCD41_CMD_START: int = 0x21B1  # start_periodic_measurement
SCD41_CMD_READ: int = 0xEC05   # read_measurement

# Timing constants — the RPi5 I2C controller requires a longer post-command
# delay than RPi4 due to different clock-stretching tolerance.
# Reducing SCD41_CMD_SLEEP_S below 0.5 causes OSError: [Errno 121] on RPi5.
SCD41_CMD_SLEEP_S: float = 0.5    # sleep after every send_command()
SCD41_STOP_SLEEP_S: float = 1.0   # extra sleep between stop and start during init
SCD41_INIT_WAIT_S: float = 6.0    # wait after start_periodic_measurement for first sample
SCD41_READ_SETTLE_S: float = 0.1  # settle time between issuing read command and reading bytes

# ---------------------------------------------------------------------------
# BME688 — temperature / humidity / pressure / gas resistance
# Connected at I2C secondary address (0x77) on /dev/i2c-1
# Note: The bme680 v2.0.0 library is used. BME688 is pin-compatible with
# BME680 for basic readings; the library works unchanged.
# ---------------------------------------------------------------------------

# bme680.I2C_ADDR_SECONDARY resolves to 0x77
BME688_I2C_ADDR_SECONDARY: int = 0x77

# ---------------------------------------------------------------------------
# SPS30 — particulate matter sensor
# I2C address 0x69, connected to /dev/i2c-1
# ---------------------------------------------------------------------------

SPS30_I2C_PORT: str = "/dev/i2c-1"
SPS30_I2C_ADDR: int = 0x69
SPS30_STARTUP_WAIT_S: float = 10.0  # stabilisation after start_measurement()
SPS30_STOP_SETTLE_S: float = 0.2    # settle time after stop_measurement() during init

# ---------------------------------------------------------------------------
# Default paths (relative to the air-monitor/ project root)
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH: str = "data/air_monitor.duckdb"
DEFAULT_LOG_PATH: str = "data/logs/pipeline.log"

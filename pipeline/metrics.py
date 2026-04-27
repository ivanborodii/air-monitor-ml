# pipeline/metrics.py
# System resource metrics collection via psutil.
# A single psutil.Process handle should be created once at startup
# and passed in on each call to avoid per-call overhead.

import logging

import psutil

logger = logging.getLogger(__name__)


def collect_system_metrics(process: psutil.Process) -> dict:
    """
    Collect a snapshot of system and process resource usage.

    Parameters
    ----------
    process : psutil.Process
        Handle to the current pipeline process (created once at startup).

    Returns
    -------
    dict with keys:
        cpu_pct          - system-wide CPU usage [%] since last call (non-blocking)
        ram_used_mb      - total system RAM used [MB]
        ram_percent      - system RAM usage [%]
        disk_used_gb     - root filesystem used [GB]
        disk_free_gb     - root filesystem free [GB]
        process_memory_mb - pipeline process RSS [MB]

    Notes
    -----
    cpu_percent(interval=None) is non-blocking and returns usage since the
    previous call. The first call in a process always returns 0.0 — this is
    acceptable and expected.

    Never raises; returns None for all fields on unexpected error.
    """
    try:
        vm = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        return {
            "cpu_pct": psutil.cpu_percent(interval=None),
            "ram_used_mb": vm.used / (1024 ** 2),
            "ram_percent": vm.percent,
            "disk_used_gb": disk.used / (1024 ** 3),
            "disk_free_gb": disk.free / (1024 ** 3),
            "process_memory_mb": process.memory_info().rss / (1024 ** 2),
        }
    except Exception as exc:
        logger.warning("System metrics collection error: %s", exc)
        return {
            "cpu_pct": None,
            "ram_used_mb": None,
            "ram_percent": None,
            "disk_used_gb": None,
            "disk_free_gb": None,
            "process_memory_mb": None,
        }

# pipeline/batching.py
# Micro-batch manager with JSON file buffer.
#
# How it works:
#   - Each incoming observation is appended to a JSON file (data/batch_buffer.json)
#     immediately after the sensor read.
#   - When the buffer reaches MICROBATCH_SIZE, the entire buffer is flushed to
#     DuckDB in a single transaction and the JSON file is truncated to [].
#   - DuckDB (raw_observations + microbatches) is only written on flush.
#   - runtime_metrics is written directly every cycle by the collector — it does
#     NOT go through the JSON buffer.
#
# Crash safety:
#   - If the pipeline crashes mid-batch, the JSON file still holds all
#     observations collected so far. On restart, BatchManager loads the JSON
#     file and continues accumulating from where it stopped.
#   - If the pipeline crashes after the DuckDB commit but before the JSON
#     truncation, batch_exists() detects the duplicate and skips re-insertion.
#
# Timestamps in each observation:
#   ts          - cycle start time
#   measured_at - exact time sensor reads began
#   loaded_at   - set at flush time (same for all rows in a batch)

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Datetime format used for JSON serialisation/deserialisation
_DT_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"


def _dt_to_str(dt: datetime) -> str:
    return dt.isoformat()


def _str_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class BatchManager:
    """
    Manages the micro-batch lifecycle using a JSON file as the write buffer.
    """

    def __init__(self, db_client, microbatch_size: int, buffer_path: str) -> None:
        """
        Parameters
        ----------
        db_client : DuckDBClient
        microbatch_size : int
            MICROBATCH_SIZE or runtime override.
        buffer_path : str
            Path to the JSON buffer file (e.g. "data/batch_buffer.json").
            Created automatically if it does not exist.
        """
        self._db = db_client
        self._size = microbatch_size
        self._buffer_path = Path(buffer_path)
        self._buffer: list[dict] = []
        self._batch_id: str = str(uuid.uuid4())

        self._load_or_reset()

        # If we restarted with a full buffer (crash during flush), flush now.
        if len(self._buffer) >= self._size:
            logger.info(
                "BatchManager: buffer is full on startup (%d records) — flushing to DuckDB",
                len(self._buffer),
            )
            try:
                self._flush()
            except Exception as exc:
                logger.error(
                    "BatchManager: startup flush failed: %s — will retry on next full batch",
                    exc,
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, observation: dict) -> tuple[str, int, bool]:
        """
        Assign batch_id and row_in_batch to the observation, append it to the
        JSON buffer, and flush to DuckDB if the buffer is now full.

        Parameters
        ----------
        observation : dict
            Must contain at least: ts (datetime), measured_at (datetime), and
            all sensor fields. Must NOT contain batch_id or row_in_batch yet.

        Returns
        -------
        (batch_id, row_in_batch, flushed) : (str, int, bool)
            flushed=True means a complete batch was written to DuckDB this call.

        Raises
        ------
        Exception
            If the JSON write fails, or if a flush is triggered and the DuckDB
            write fails. The caller (collector) catches this and marks the
            cycle as failed.
        """
        row_in_batch = len(self._buffer)
        batch_id = self._batch_id

        # Serialise datetimes to ISO strings for JSON storage
        serialised = _serialise_observation(observation)
        serialised["batch_id"] = batch_id
        serialised["row_in_batch"] = row_in_batch

        self._buffer.append(serialised)
        self._save_buffer()

        if len(self._buffer) >= self._size:
            self._flush()
            return batch_id, row_in_batch, True

        return batch_id, row_in_batch, False

    @property
    def current_batch_id(self) -> str:
        return self._batch_id

    @property
    def current_buffer_size(self) -> int:
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_or_reset(self) -> None:
        """
        Load an existing JSON buffer from disk (crash recovery), or start fresh.
        """
        self._buffer_path.parent.mkdir(parents=True, exist_ok=True)

        if self._buffer_path.exists() and self._buffer_path.stat().st_size > 2:
            try:
                data = json.loads(self._buffer_path.read_text(encoding="utf-8"))
                if isinstance(data, list) and len(data) > 0:
                    self._buffer = data
                    # Recover batch_id from the buffered observations
                    self._batch_id = data[0].get("batch_id", str(uuid.uuid4()))
                    logger.info(
                        "BatchManager: recovered %d observations from %s (batch=%s...)",
                        len(data),
                        self._buffer_path,
                        self._batch_id[:8],
                    )
                    return
            except Exception as exc:
                logger.warning(
                    "BatchManager: could not read buffer file %s: %s — starting fresh",
                    self._buffer_path,
                    exc,
                )

        logger.info(
            "BatchManager: starting fresh batch %s...", self._batch_id[:8]
        )
        self._truncate_buffer()

    def _flush(self) -> None:
        """
        Bulk-insert all buffered observations into DuckDB and truncate the JSON file.

        Uses an explicit transaction so either all rows land in DuckDB or none do.
        The JSON file is only truncated after a successful commit.
        """
        if not self._buffer:
            return

        batch_id = self._batch_id
        buffer = list(self._buffer)
        loaded_at = datetime.now(timezone.utc)

        # --- Deduplication guard ---
        # If this batch is already in DuckDB (crash between commit and truncate),
        # skip re-insertion and just clear the JSON file.
        if self._db.batch_exists(batch_id):
            logger.warning(
                "BatchManager: batch %s... already in DuckDB — skipping duplicate flush",
                batch_id[:8],
            )
            self._buffer = []
            self._batch_id = str(uuid.uuid4())
            self._truncate_buffer()
            logger.info("BatchManager: opened new batch %s...", self._batch_id[:8])
            return

        # --- DuckDB transaction ---
        self._db.begin()
        try:
            start_ts = _str_to_dt(buffer[0]["measured_at"])
            end_ts = _str_to_dt(buffer[-1]["measured_at"])

            self._db.insert_batch_record(batch_id, start_ts, end_ts, len(buffer), loaded_at)

            for raw in buffer:
                obs = _deserialise_observation(raw)
                obs["loaded_at"] = loaded_at
                self._db.insert_observation(obs)

            self._db.commit()
        except Exception as exc:
            self._db.rollback()
            logger.error(
                "BatchManager: DuckDB flush failed for batch %s...: %s",
                batch_id[:8],
                exc,
            )
            raise

        # --- Truncate JSON only after successful commit ---
        self._buffer = []
        self._batch_id = str(uuid.uuid4())
        self._truncate_buffer()

        logger.info(
            "BatchManager: flushed batch %s... (%d records) to DuckDB, loaded_at=%s",
            batch_id[:8],
            len(buffer),
            loaded_at.isoformat(),
        )
        logger.info("BatchManager: opened new batch %s...", self._batch_id[:8])

    def _save_buffer(self) -> None:
        """Overwrite the JSON buffer file with current in-memory state."""
        self._buffer_path.write_text(
            json.dumps(self._buffer, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _truncate_buffer(self) -> None:
        """Reset the JSON buffer file to an empty list."""
        self._buffer_path.write_text("[]", encoding="utf-8")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialise_observation(obs: dict) -> dict:
    """
    Convert a raw observation dict (with datetime objects) to a JSON-safe dict
    by converting all datetime values to ISO-format strings.
    """
    result = {}
    for k, v in obs.items():
        if isinstance(v, datetime):
            result[k] = v.isoformat()
        else:
            result[k] = v
    return result


def _deserialise_observation(raw: dict) -> dict:
    """
    Convert a JSON-loaded observation dict back to Python types.
    Datetime fields (ts, measured_at) are parsed from ISO strings.
    """
    result = dict(raw)
    for field in ("ts", "measured_at"):
        if field in result and isinstance(result[field], str):
            result[field] = _str_to_dt(result[field])
    return result

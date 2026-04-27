"""Save and load trained models with joblib.

Models are stored in MODELS_DIR as:
    {name}_{timestamp}.pkl        — compressed joblib dump
    {name}_{timestamp}_meta.json  — training metadata

load_latest_model() always returns the most recently saved version
of a given model name (alphabetically last timestamp).
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from ml.config import MODELS_DIR


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_model(model: Any, name: str, metadata: dict) -> Path:
    """Persist model to disk and write companion metadata JSON.

    Returns the path to the saved .pkl file.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    ts = _ts()
    model_path = Path(MODELS_DIR) / f"{name}_{ts}.pkl"
    meta_path  = Path(MODELS_DIR) / f"{name}_{ts}_meta.json"

    joblib.dump(model, model_path, compress=3)
    metadata = dict(metadata)
    metadata["model_size_kb"] = round(model_path.stat().st_size / 1024, 2)
    metadata["model_name"] = name
    metadata["saved_at"]   = ts
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    return model_path


def load_latest_model(name: str) -> tuple[Any, dict]:
    """Load the most recently saved model with the given name.

    Returns (model, metadata_dict).
    Raises FileNotFoundError when no matching file exists.
    """
    pattern = str(Path(MODELS_DIR) / f"{name}_*.pkl")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No saved model found for '{name}' in {MODELS_DIR!r}. "
            "Run training first."
        )
    model_path = matches[-1]
    meta_path  = model_path.replace(".pkl", "_meta.json")

    model    = joblib.load(model_path)
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            metadata = json.load(fh)

    return model, metadata


def list_models() -> list[dict]:
    """Return metadata for all saved models, sorted by name + save time."""
    pattern = str(Path(MODELS_DIR) / "*_meta.json")
    results = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            results.append(json.load(fh))
    return results

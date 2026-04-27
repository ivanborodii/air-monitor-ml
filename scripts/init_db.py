#!/usr/bin/env python3
# scripts/init_db.py
# One-time database initialisation script.
# Run once before starting the pipeline for the first time:
#
#   cd air-monitor
#   python scripts/init_db.py [--config config/settings.yaml]
#
# Safe to re-run: all CREATE TABLE statements use IF NOT EXISTS.

import argparse
import sys
from pathlib import Path

# Allow running from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import duckdb
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Initialise the air-monitor DuckDB schema.")
    parser.add_argument(
        "--config",
        default="config/settings.yaml",
        help="Path to settings.yaml (default: config/settings.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    db_path = config.get("db_path", "data/air_monitor.duckdb")

    # Resolve schema.sql relative to this script's location
    schema_path = Path(__file__).resolve().parent.parent / "storage" / "schema.sql"

    # Ensure data directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    (Path(db_path).parent.parent / "data" / "logs").mkdir(parents=True, exist_ok=True)

    schema_sql = schema_path.read_text()

    conn = duckdb.connect(db_path)
    try:
        conn.execute(schema_sql)
        conn.commit()
        print(f"Schema initialised at: {db_path}")
        # Show created tables
        tables = conn.execute("SHOW TABLES").fetchall()
        for (table,) in tables:
            print(f"  table: {table}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()

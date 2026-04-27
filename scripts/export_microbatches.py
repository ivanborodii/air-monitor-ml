"""Export raw_observations from DuckDB to a CSV file for ML training.

Usage
-----
From the air-monitor/ directory:

    python scripts/export_microbatches.py
    python scripts/export_microbatches.py --db data/air_monitor.duckdb --out data/microbatches.csv
"""

import argparse

import duckdb
import pandas as pd

from config.constants import DEFAULT_DB_PATH

DEFAULT_OUT = "data/microbatches.csv"


def run(db_path: str, out_path: str) -> None:
    con = duckdb.connect(db_path, read_only=True)
    try:
        df = con.execute(
            "SELECT * FROM raw_observations ORDER BY batch_id, row_in_batch"
        ).df()
    finally:
        con.close()

    n_batches = df["batch_id"].nunique() if "batch_id" in df.columns else 0
    df.to_csv(out_path, index=False)
    print(f"Exported {len(df)} observations ({n_batches} batches) → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DuckDB raw_observations to CSV")
    parser.add_argument("--db",  default=DEFAULT_DB_PATH, help="Path to DuckDB file")
    parser.add_argument("--out", default=DEFAULT_OUT,     help="Output CSV path")
    args = parser.parse_args()
    run(args.db, args.out)


if __name__ == "__main__":
    main()

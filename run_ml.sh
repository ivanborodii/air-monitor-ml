#!/usr/bin/env bash
# run_ml.sh - train all ML models and run inference on the latest batch
# Run from the air-monitor/ directory:
#   bash run_ml.sh
#   bash run_ml.sh --system              # also include cpu_pct / ram_percent as features
#   bash run_ml.sh --system --temporal   # + hour_sin/cos, day_of_week, rolling mean features

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate the project virtual environment
VENV="$SCRIPT_DIR/../.venv/bin/activate"
if [[ ! -f "$VENV" ]]; then
    echo "ERROR: virtualenv not found at $VENV"
    exit 1
fi
# shellcheck source=/dev/null
source "$VENV"

EXTRA_FLAGS=""
TEMPORAL_FLAG=""
for arg in "$@"; do
    case "$arg" in
        --system)   EXTRA_FLAGS="--system"   ;;
        --temporal) TEMPORAL_FLAG="--temporal" ;;
    esac
done
[[ -n "$EXTRA_FLAGS"   ]] && echo "[info] System metrics (cpu_pct, ram_percent) included as features."
[[ -n "$TEMPORAL_FLAG" ]] && echo "[info] Temporal + rolling features enabled."

# ---------------------------------------------------------------------------
# 0. Clear previous outputs
# ---------------------------------------------------------------------------
echo ""
echo "==> Clearing previous outputs..."
rm -f outputs/models/* outputs/reports/* outputs/predictions/*
echo "    Done."

# ---------------------------------------------------------------------------
# 1. Check DB is accessible (not locked by DBeaver or the pipeline)
# ---------------------------------------------------------------------------
echo ""
echo "==> Checking database access..."
python - <<'EOF'
import sys, duckdb
try:
    con = duckdb.connect("data/air_monitor.duckdb", read_only=True)
    n = con.execute("SELECT COUNT(*) FROM raw_observations").fetchone()[0]
    batches = con.execute("SELECT COUNT(DISTINCT batch_id) FROM raw_observations").fetchone()[0]
    con.close()
    print(f"    OK - {n} observations, {batches} batches available")
except Exception as e:
    print(f"    ERROR: {e}")
    print("    Is DBeaver still connected? Disconnect it and re-run.")
    sys.exit(1)
EOF

# ---------------------------------------------------------------------------
# 2. Train classification models
# ---------------------------------------------------------------------------
echo ""
echo "==> Training classification models..."
python -m ml.train_classification --db data/air_monitor.duckdb $EXTRA_FLAGS $TEMPORAL_FLAG

# ---------------------------------------------------------------------------
# 3. Train anomaly detection models
# ---------------------------------------------------------------------------
echo ""
echo "==> Training anomaly detection models..."
python -m ml.train_anomaly --db data/air_monitor.duckdb $EXTRA_FLAGS $TEMPORAL_FLAG

# ---------------------------------------------------------------------------
# 4. Run inference on the latest batch
# ---------------------------------------------------------------------------
echo ""
echo "==> Preparing latest batch for inference..."
BUFFER="data/batch_buffer.json"
INFERENCE_INPUT="data/latest_microbatch.json"

if [[ ! -f "$BUFFER" ]]; then
    echo "    WARNING: $BUFFER not found - skipping inference."
    echo "    Start the pipeline first to collect a batch."
else
    BUFFER_SIZE=$(python -c "import json; d=json.load(open('$BUFFER')); print(len(d))")
    if [[ "$BUFFER_SIZE" -eq 0 ]]; then
        echo "    WARNING: $BUFFER is empty - skipping inference."
        echo "    Wait for the pipeline to collect at least one observation."
    else
        cp "$BUFFER" "$INFERENCE_INPUT"
        echo "    Copied $BUFFER ($BUFFER_SIZE observations) -> $INFERENCE_INPUT"
        echo ""
        echo "==> Running inference..."
        python -m ml.inference --input "$INFERENCE_INPUT"
    fi
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "==> All done."
echo "    Reports   : outputs/reports/"
echo "    Models    : outputs/models/"
echo "    Predictions: outputs/predictions/"

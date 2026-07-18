#!/bin/bash
# Run the full evaluation pipeline over whatever synthesised clips currently exist:
# score every evaluable (model, accent, speaker, utterance) cell, then regenerate the
# summary tables and figures. Re-runnable and path-independent.
#
#   bash run_pipeline.sh
#   PYTHON=/path/to/python bash run_pipeline.sh        # choose the interpreter
#   EVAL_SKIP_UTMOS=1 bash run_pipeline.sh             # defer the slow UTMOS pass
#
# PYTHON must be the main evaluation interpreter (pandas + the native metrics); it shells
# out to the isolated UTMOS / GenAID environments (see UTMOS_PYTHON / GENAID_PYTHON). It
# defaults to the repo-local .conda interpreter. Backfill a deferred metric afterwards with
# backfill_metrics.py.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
PYTHON=${PYTHON:-"$REPO/.conda/bin/python"}
cd "$HERE"

echo "=== scoring (run_eval.py) ==="
"$PYTHON" run_eval.py

echo "=== summaries + figures (visualize_results.py) ==="
"$PYTHON" visualize_results.py

echo "=== pipeline complete ==="

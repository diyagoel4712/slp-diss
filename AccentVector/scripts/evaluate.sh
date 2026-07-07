#!/bin/bash
# Score the alpha sweep with the repo's eval suite -> one metrics row per alpha.
# Runs on the Mac (eval env), no GPU needed. Accent metrics need GenAID's
# isolated env configured in evaluation_functions.py.
set -euo pipefail

ACCENT_DIR=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$ACCENT_DIR:${PYTHONPATH:-}"

ACCENT_NAME=${ACCENT_NAME:-british}
SWEEP_DIR=${SWEEP_DIR:-"$ACCENT_DIR/results/${ACCENT_NAME}"}
TRANSCRIPTS=${TRANSCRIPTS:-"$ACCENT_DIR/transcripts/eval_transcripts.txt"}
REF_WAV=${REF_WAV:-"$ACCENT_DIR/refs/neutral.wav"}
ACCENT_REF=${ACCENT_REF:-}          # dir of real target-accent clips (for cs_accent)
TARGET_ACCENT=${TARGET_ACCENT:-English}
OUT_CSV=${OUT_CSV:-"$SWEEP_DIR/metrics.csv"}

python -m accent_vector.evaluate \
    --sweep-dir "$SWEEP_DIR" \
    --transcripts "$TRANSCRIPTS" \
    --ref-wav "$REF_WAV" \
    ${ACCENT_REF:+--accent-ref "$ACCENT_REF"} \
    --target-accent "$TARGET_ACCENT" \
    --out-csv "$OUT_CSV"

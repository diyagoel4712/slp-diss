#!/bin/bash
# Extract the accent vector: tau = theta_ft - theta_pre  (paper Eq. 1-3).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

ACCENT_DIR=$(cd "$(dirname "$0")/.." && pwd)
F5_ROOT=${F5_ROOT:-"$ACCENT_DIR/../Expressive-Vectors/F5-TTS"}
export PYTHONPATH="$F5_ROOT/src:$ACCENT_DIR:${PYTHONPATH:-}"

ACCENT_NAME=${ACCENT_NAME:-british}
PRETRAIN=${PRETRAIN:-"$F5_ROOT/ckpts/F5TTS_v1_Base/model_1250000.pt"}
FINETUNED=${FINETUNED:-"$F5_ROOT/ckpts/${ACCENT_NAME}/model_60000.pt"}
OUT=${OUT:-"$ACCENT_DIR/vectors/${ACCENT_NAME}.pt"}

python -m accent_vector.extract_vector extract \
    --pretrained "$PRETRAIN" \
    --finetuned "$FINETUNED" \
    --out "$OUT"

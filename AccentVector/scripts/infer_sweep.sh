#!/bin/bash
# Alpha sweep: synthesize held-out English transcripts at increasing accent
# strength (paper Eq. 4 / Figure 3), with the accent's native-language (L1) reference
# held FIXED across alpha so the vector is the only thing varying (alpha=0 = pretrained
# model cloning the reference; alpha=1 = fully fine-tuned).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

ACCENT_DIR=$(cd "$(dirname "$0")/.." && pwd)
F5_ROOT=${F5_ROOT:-"$ACCENT_DIR/../F5-TTS"}
export PYTHONPATH="$F5_ROOT/src:$ACCENT_DIR:${PYTHONPATH:-}"

ACCENT_NAME=${ACCENT_NAME:-british}
PRETRAIN=${PRETRAIN:-"$F5_ROOT/ckpts/F5TTS_v1_Base/model_1250000.pt"}
VECTOR=${VECTOR:-"$ACCENT_DIR/vectors/${ACCENT_NAME}.pt"}
ALPHAS=${ALPHAS:-"0,0.2,0.4,0.6,0.8,1.0"}
# native-language (L1) reference for this accent; REF_TEXT must be its transcript
REF_AUDIO=${REF_AUDIO:-"$ACCENT_DIR/refs/england.wav"}
REF_TEXT=${REF_TEXT:-"Some call me nature, others call me mother nature."}
TRANSCRIPTS=${TRANSCRIPTS:-"$ACCENT_DIR/transcripts/eval_transcripts.txt"}
OUT_DIR=${OUT_DIR:-"$ACCENT_DIR/results/${ACCENT_NAME}"}

python -m accent_vector.infer_accent \
    --pretrained "$PRETRAIN" \
    --vector "$VECTOR" \
    --alphas "$ALPHAS" \
    --ref-audio "$REF_AUDIO" \
    --ref-text "$REF_TEXT" \
    --transcripts "$TRANSCRIPTS" \
    --out-dir "$OUT_DIR"

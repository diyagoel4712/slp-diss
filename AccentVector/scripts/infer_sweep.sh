#!/bin/bash
# Alpha sweep: synthesize held-out English transcripts at increasing accent
# strength (paper Eq. 4 / Figure 3), with the accent's native-language (L1) reference
# held FIXED across alpha so the vector is the only thing varying (alpha=0 = pretrained
# model cloning the reference; alpha=1 = fully fine-tuned).
#
# Two tracks (see infer_accent.py):
#   LORA=1 (default, paper-matching)  native LoRA sweep: build base+LoRA once,
#           rescale the branch by alpha in place -- exact theta_pre + alpha*theta_LoRA,
#           no merge. VECTOR is a LoRA vector/snapshot (lora_state_dict); needs the
#           training run's CONFIG (config.yaml) and VOCAB (vocab.txt).
#   LORA=0  merged full-weight sweep: compose theta_pre + alpha*tau into a checkpoint
#           per alpha. VECTOR is a full-weight diff (extract_vector extract). Only for
#           FULL fine-tunes -- NOT the unmerged-LoRA vectors this project produces.
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

LORA=${LORA:-1}   # 1 = native LoRA sweep (paper-matching, default); 0 = merged full-weight sweep

ARGS=(
    --pretrained "$PRETRAIN"
    --alphas "$ALPHAS"
    --ref-audio "$REF_AUDIO"
    --ref-text "$REF_TEXT"
    --transcripts "$TRANSCRIPTS"
    --out-dir "$OUT_DIR"
)

if [ "$LORA" = "1" ]; then
    # config.yaml + vocab.txt from the training run dir (both saved next to ckpts/).
    CONFIG=${CONFIG:?LORA=1 needs CONFIG=<run_dir>/config.yaml from the training run}
    VOCAB=${VOCAB:?LORA=1 needs VOCAB=<run_dir>/vocab.txt from the training run}
    ARGS+=(--lora --lora-vector "$VECTOR" --config "$CONFIG" --vocab "$VOCAB")
    # single-accent runs (lora_feature_dim=null) ignore the branch idx; only set
    # these for a multi-accent model. Resolve name->idx via lora_mapping.json, or
    # pass LORA_IDX directly. (if/then, not `&&`: an empty-var test returns non-zero
    # and would abort under `set -e`.)
    if [ -n "${LORA_LABEL:-}" ];   then ARGS+=(--lora-label "$LORA_LABEL"); fi
    if [ -n "${LORA_MAPPING:-}" ]; then ARGS+=(--lora-mapping "$LORA_MAPPING"); fi
    if [ -n "${LORA_IDX:-}" ];     then ARGS+=(--lora-idx "$LORA_IDX"); fi
else
    ARGS+=(--vector "$VECTOR")
fi

python -m accent_vector.infer_accent "${ARGS[@]}"

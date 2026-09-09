#!/bin/bash
# Alpha sweep: synthesize held-out English transcripts at increasing accent
# strength (paper Eq. 4 / Figure 3), with the accent's native-language (L1) reference
# held FIXED across alpha so the vector is the only thing varying (alpha=0 = pretrained
# model cloning the reference; alpha=1 = fully fine-tuned).
#
# The sweep is native LoRA: build base+LoRA once and rescale the branch by alpha in
# place -- exact theta_pre + alpha*theta_LoRA, no merge. VECTOR is a LoRA
# vector/snapshot (lora_state_dict); CONFIG (config.yaml) and VOCAB (vocab.txt) come
# from the training run so the LoRA architecture matches the vector.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

ACCENT_DIR=$(cd "$(dirname "$0")/../.." && pwd)
F5_ROOT=${F5_ROOT:-"$ACCENT_DIR/../F5-TTS"}
export PYTHONPATH="$F5_ROOT/src:$ACCENT_DIR/src:${PYTHONPATH:-}"

ACCENT_NAME=${ACCENT_NAME:-british}
PRETRAIN=${PRETRAIN:-"$F5_ROOT/ckpts/F5TTS_v1_Base/model_1250000.pt"}
VECTOR=${VECTOR:-"$ACCENT_DIR/vectors/${ACCENT_NAME}.pt"}
ALPHAS=${ALPHAS:-"0,0.25,0.5,0.75,1.0"}
# native-language (L1) reference for this accent; REF_TEXT must be its transcript
REF_AUDIO=${REF_AUDIO:-"$ACCENT_DIR/refs/england.wav"}
REF_TEXT=${REF_TEXT:-"Some call me nature, others call me mother nature."}
TRANSCRIPTS=${TRANSCRIPTS:-"$ACCENT_DIR/data/transcripts/eval_transcripts.txt"}
OUT_DIR=${OUT_DIR:-"$ACCENT_DIR/results/per-accent/${ACCENT_NAME}"}
# transcript sharding for multi-GPU fan-out: render only indices == SHARD_INDEX (mod
# SHARD_COUNT), keeping the global utt#### name so shards reassemble into one alpha_<a>/.
SHARD_INDEX=${SHARD_INDEX:-0}
SHARD_COUNT=${SHARD_COUNT:-1}


ARGS=(
    --pretrained "$PRETRAIN"
    --alphas "$ALPHAS"
    --ref-audio "$REF_AUDIO"
    --ref-text "$REF_TEXT"
    --transcripts "$TRANSCRIPTS"
    --out-dir "$OUT_DIR"
    --shard-index "$SHARD_INDEX"
    --shard-count "$SHARD_COUNT"
)

# config.yaml + vocab.txt from the training run dir (both saved next to ckpts/):
# the LoRA architecture (rank, target modules) must match the trained vector.
CONFIG=${CONFIG:?needs CONFIG=<run_dir>/config.yaml from the training run}
VOCAB=${VOCAB:?needs VOCAB=<run_dir>/vocab.txt from the training run}
ARGS+=(--lora-vector "$VECTOR" --config "$CONFIG" --vocab "$VOCAB")
# single-accent runs (lora_feature_dim=null) ignore the branch idx; only set these
# for a multi-accent model. Resolve name->idx via lora_mapping.json, or pass
# LORA_IDX directly. (if/then, not `&&`: an empty-var test returns non-zero and
# would abort under `set -e`.)
if [ -n "${LORA_LABEL:-}" ];   then ARGS+=(--lora-label "$LORA_LABEL"); fi
if [ -n "${LORA_MAPPING:-}" ]; then ARGS+=(--lora-mapping "$LORA_MAPPING"); fi
if [ -n "${LORA_IDX:-}" ];     then ARGS+=(--lora-idx "$LORA_IDX"); fi

python -m accent_vector.infer_accent "${ARGS[@]}"

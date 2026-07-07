#!/bin/bash
# Phase A fine-tune: F5-TTS on VCTK England speech -> theta_ft (the checkpoint we
# later diff against theta_pre to get the accent vector).
#
# Mirrors Expressive-Vectors' finetuning_model.sh: full fine-tune with the plain
# F5TTS_v1_Base recipe. The task-vector math (theta_ft - theta_pre) is identical
# whether the fine-tune is full or LoRA; we use full FT because that is the path
# the upstream repo actually exercises end-to-end. To match the paper's LoRA
# (rank 16, all linear layers, lr 3e-5) instead, swap in F5TTS_v1_LoRA and verify
# the checkpoint stores MERGED weights before extracting (see README).
#
# Edit the paths below for your machine, then: bash scripts/finetune.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

ACCENT_DIR=$(cd "$(dirname "$0")/.." && pwd)
F5_ROOT=${F5_ROOT:-"$ACCENT_DIR/../Expressive-Vectors/F5-TTS"}
export PYTHONPATH="$F5_ROOT/src:$ACCENT_DIR:${PYTHONPATH:-}"

ACCENT_NAME=${ACCENT_NAME:-british}          # dataset/label
VCTK_ACCENT=${VCTK_ACCENT:-English}          # VCTK ACCENTS value (England English)
VCTK_ROOT=${VCTK_ROOT:-/data/VCTK-Corpus-0.92}
PRETRAIN=${PRETRAIN:-"$F5_ROOT/ckpts/F5TTS_v1_Base/model_1250000.pt"}

DATA_DIR="$F5_ROOT/data/${ACCENT_NAME}_pinyin"   # finetune_cli expects <name>_pinyin
META_CSV="$ACCENT_DIR/data/${ACCENT_NAME}/metadata.csv"

# 1. VCTK -> metadata.csv (England speakers, dur >= 3 s)
python -m accent_vector.data_preprocess build-vctk \
    --vctk-root "$VCTK_ROOT" \
    --accent "$VCTK_ACCENT" \
    --out-csv "$META_CSV"

# 2. metadata.csv -> F5 Arrow dataset
python -m accent_vector.data_preprocess prepare \
    --metadata "$META_CSV" \
    --audio-root "$VCTK_ROOT" \
    --out-dir "$DATA_DIR"

# 3. Fine-tune F5-TTS (theta_ft). Checkpoints land under F5's ckpts/<name>/.
python "$F5_ROOT/src/f5_tts/train/finetune_cli.py" \
    --finetune \
    --pretrain "$PRETRAIN" \
    --dataset_name "$ACCENT_NAME"

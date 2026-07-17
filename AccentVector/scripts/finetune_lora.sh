#!/bin/bash
# LoRA fine-tune F5-TTS on ONE accent -> one accent vector (tau = the LoRA weights).
# Run once per accent. The fork's finetune_cli.py is Hydra-based, so overrides use
# key=value syntax (NOT --flags).
#
#   ACCENT_NAME=british \
#   METADATA_CSV=/path/to/british.csv AUDIO_ROOT=/path/to/audio \
#   PRETRAIN=$F5_ROOT/ckpts/F5TTS_v1_Base/model_1250000.pt \
#   bash scripts/finetune_lora.sh
#
# Stop cleanly when the samples sound right:  touch <save_dir>/STOP
# (save_dir is printed at launch; it finishes the step, writes model_last.pt, exits.)
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

ACCENT_DIR=$(cd "$(dirname "$0")/.." && pwd)
F5_ROOT=${F5_ROOT:-"$ACCENT_DIR/../F5-TTS"}
export PYTHONPATH="$F5_ROOT/src:$ACCENT_DIR:${PYTHONPATH:-}"

ACCENT_NAME=${ACCENT_NAME:?set ACCENT_NAME (e.g. british)}
PRETRAIN=${PRETRAIN:-"$F5_ROOT/ckpts/F5TTS_v1_Base/model_1250000.pt"}
LORA_LABEL=${LORA_LABEL:-0}                       # single accent -> one label
DATA_DIR="$F5_ROOT/data/${ACCENT_NAME}_pinyin"    # finetune_cli expects <name>_pinyin/

# 1. metadata.csv -> F5 Arrow dataset WITH a constant lora_label (skip if present).
if [ ! -f "$DATA_DIR/raw.arrow" ]; then
    METADATA_CSV=${METADATA_CSV:?dataset not prepared; set METADATA_CSV=<audio_file|text csv>}
    AUDIO_ROOT=${AUDIO_ROOT:?set AUDIO_ROOT (prefix joined to CSV audio paths)}
    python -m accent_vector.data_preprocess prepare \
        --metadata "$METADATA_CSV" \
        --audio-root "$AUDIO_ROOT" \
        --out-dir "$DATA_DIR" \
        --lora-label "$LORA_LABEL"
fi

# 2. lora_mapping.json (label name -> idx). finetune_cli copies it; inference reads it.
if [ ! -f "$DATA_DIR/lora_mapping.json" ]; then
    printf '{"%s": %s}\n' "$ACCENT_NAME" "$LORA_LABEL" > "$DATA_DIR/lora_mapping.json"
fi

# 3. LoRA fine-tune (Hydra overrides). Snapshots -> <save_dir>/ckpts/snapshots/lora_<step>.pt
python "$F5_ROOT/src/f5_tts/train/finetune_cli.py" \
    --config-name F5TTS_v1_LoRA_accent \
    datasets.name="$ACCENT_NAME" \
    ckpts.pretrain="$PRETRAIN" \
    ckpts.wandb_run_name="lora_${ACCENT_NAME}" \
    "$@"

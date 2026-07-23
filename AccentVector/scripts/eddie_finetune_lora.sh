#!/bin/bash
# Eddie (SGE) GPU wrapper for scripts/finetune_lora.sh -- LoRA fine-tune one accent.
#   cd ~/slp-diss/AccentVector && mkdir -p logs
#   qsub scripts/eddie_finetune_lora.sh
# Override any env var at submit time, e.g.:
#   qsub -v ACCENT_NAME=dutch scripts/eddie_finetune_lora.sh
#
#$ -N ft_dutch
#$ -cwd
#$ -q gpu
#$ -pe gpu-a100 1            # <-- VERIFY: GPU parallel-env/queue for your allocation
#$ -l h_rt=24:00:00         #     (you have slpgpustorage -- check your group's GPU queue)
#$ -l h_vmem=64G
#$ -o logs/ft.$JOB_ID.out
#$ -e logs/ft.$JOB_ID.err

set -euo pipefail

# --- environment (adjust module names to your Eddie setup) ---
. /etc/profile.d/modules.sh
module load cuda            # <-- match the CUDA your torch build expects
module load anaconda
conda activate f5-tts       # your F5-TTS env

# compute nodes have no internet -> keep wandb from hanging
export WANDB_MODE=${WANDB_MODE:-offline}

ACCENT_DIR=$(cd "$(dirname "$0")/.." && pwd)

# --- run config (all overridable via `qsub -v KEY=VAL,...`) ---
export F5_ROOT=${F5_ROOT:-"$ACCENT_DIR/../F5-TTS"}          # confirm F5-TTS location
export ACCENT_NAME=${ACCENT_NAME:-dutch}
export METADATA_CSV=${METADATA_CSV:-/exports/eddie/scratch/s2247837/data/cgn_dutch_clips/metadata.dnsmos.csv}
export AUDIO_ROOT=${AUDIO_ROOT:-/exports/eddie/scratch/s2247837/data/cgn_dutch_clips}
export PRETRAIN=${PRETRAIN:-"$F5_ROOT/ckpts/F5TTS_v1_Base/model_1250000.pt"}
export CKPT_ROOT=${CKPT_ROOT:-/exports/eddie/scratch/s2247837/accentvector-exps}
export LORA_LABEL=${LORA_LABEL:-0}
# base vocab, without staging a file into the F5-TTS tree (prepare reads F5_VOCAB)
export F5_VOCAB=${F5_VOCAB:-"$F5_ROOT/examples/vocab.txt"}

echo "accent=$ACCENT_NAME  data=$METADATA_CSV"
echo "F5_ROOT=$F5_ROOT  CKPT_ROOT=$CKPT_ROOT"
nvidia-smi -L || true

# builds data/<accent>_pinyin (prepare) if needed, then LoRA fine-tunes
bash "$ACCENT_DIR/scripts/finetune_lora.sh"

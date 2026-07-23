#!/bin/bash
# Eddie (SGE) GPU wrapper for scripts/finetune_lora.sh -- LoRA fine-tune one accent.
#   cd /exports/chss/eddie/ppls/groups/slpgpustorage/users/s2247837/slp-diss/AccentVector && mkdir -p logs
#   qsub scripts/eddie_finetune_lora.sh
# Override any env var at submit time, e.g.:
#   qsub -v ACCENT_NAME=dutch scripts/eddie_finetune_lora.sh
# The job name is static (SGE parses -N before the script runs, so it can't read
# ACCENT_NAME); override it on the command line to match, e.g.:
#   qsub -N ft_dutch -v ACCENT_NAME=dutch scripts/eddie_finetune_lora.sh
#
#$ -N ft_accent
#$ -cwd
#$ -q gpu
#$ -l gpu=1            
#$ -l h_rt=12:00:00         
#$ -l h_vmem=32G
#$ -o logs/ft.$JOB_ID.out
#$ -e logs/ft.$JOB_ID.err
#$ -P ppls_slpgpu
#$ -M s2247837@ed.ac.uk

set -euo pipefail

# --- environment (adjust module names to your Eddie setup) ---
# No `module load cuda`: the f5-tts torch wheel bundles its own CUDA runtime and
# uses the node's driver directly. Loading a system CUDA can shadow those libs.
. /etc/profile.d/modules.sh
module load anaconda
conda activate f5-tts       # your F5-TTS env

# compute nodes have no internet. wandb would need a cached API key (else the
# trainer silently disables logging) and only writes local files to sync later,
# so default to tensorboard: fully local, no key, event files under runs/lora_<accent>.
export WANDB_MODE=${WANDB_MODE:-offline}
export LOGGER=${LOGGER:-tensorboard}
# no internet on the node: never let a stray HF call hang the GPU -- fail fast.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# SGE runs a spooled COPY of this script, so $0 is not the scripts/ path. With
# -cwd the submission dir (AccentVector) is SGE_O_WORKDIR; fall back to $PWD when
# run outside SGE (e.g. testing locally from AccentVector/).
ACCENT_DIR="${SGE_O_WORKDIR:-$PWD}"

# --- run config (all overridable via `qsub -v KEY=VAL,...`) ---
export F5_ROOT=${F5_ROOT:-"$ACCENT_DIR/../F5-TTS"}          # confirm F5-TTS location
export ACCENT_NAME=${ACCENT_NAME:-dutch}
export METADATA_CSV=${METADATA_CSV:-/exports/eddie/scratch/s2247837/data/cgn_dutch_clips/metadata.dnsmos.csv}
export AUDIO_ROOT=${AUDIO_ROOT:-/exports/eddie/scratch/s2247837/data/cgn_dutch_clips}
export PRETRAIN=${PRETRAIN:-/exports/eddie/scratch/s2247837/ckpts/F5TTS_v1_Base/model_1250000.safetensors}
export CKPT_ROOT=${CKPT_ROOT:-/exports/eddie/scratch/s2247837/accentvector-exps}
export LORA_LABEL=${LORA_LABEL:-0}
# base vocab, without staging a file into the F5-TTS tree (prepare reads F5_VOCAB)
export F5_VOCAB=${F5_VOCAB:-"$F5_ROOT/examples/vocab.txt"}
# vocoder for log_samples: pre-downloaded to scratch so the offline node uses it
# locally (dir must hold config.yaml + pytorch_model.bin). NUM_WORKERS defaults to
# 1 -- the gpu=1 allocation grants 1 CPU core, so more workers just oversubscribe.
VOCODER_DIR=${VOCODER_DIR:-/exports/eddie/scratch/s2247837/vocos-mel-24khz}
NUM_WORKERS=${NUM_WORKERS:-1}

echo "accent=$ACCENT_NAME  data=$METADATA_CSV"
echo "F5_ROOT=$F5_ROOT  CKPT_ROOT=$CKPT_ROOT"
nvidia-smi -L || true
# confirm torch sees its bundled CUDA + the GPU (no system CUDA module needed)
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

# record run provenance (code versions + config) -> log + CKPT_ROOT/provenance/.
# Best-effort: a provenance hiccup must never kill the GPU job.
bash "$ACCENT_DIR/scripts/record_provenance.sh" "$ACCENT_DIR" "$F5_ROOT" "$CKPT_ROOT/provenance" \
    || echo "warning: provenance capture failed (continuing)"

# builds data/<accent>_pinyin (prepare) if needed, then LoRA fine-tunes.
# Extra args are forwarded to finetune_cli.py as Hydra overrides.
bash "$ACCENT_DIR/scripts/finetune_lora.sh" \
    ckpts.logger="$LOGGER" \
    datasets.num_workers="$NUM_WORKERS" \
    model.vocoder.is_local=True \
    model.vocoder.local_path="$VOCODER_DIR"

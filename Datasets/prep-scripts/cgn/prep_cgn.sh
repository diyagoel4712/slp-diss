#!/bin/bash
# Eddie (SGE) batch job: build the Dutch/CGN F5-TTS dataset. CPU-only.
#   qsub Datasets/scripts/cgn/prep_cgn.sh
#
#$ -N cgn_prep
#$ -cwd
#$ -l h_rt=06:00:00
#$ -l h_vmem=16G
#$ -o logs/cgn_prep.$JOB_ID.out
#$ -e logs/cgn_prep.$JOB_ID.err

set -euo pipefail

# --- environment (adjust these two lines to your Eddie setup) ---
. /etc/profile.d/modules.sh
module load anaconda
source activate f5-tts          # <-- your F5-TTS conda env name

SCRATCH=/exports/eddie/scratch/s2247837/data
ROOT=$SCRATCH/cgn_dutch          # staged CGN: audio/wav + annot/text/ort
CLIPS=$SCRATCH/cgn_dutch_clips   # cut utterance clips + metadata.csv
DS=$SCRATCH/cgn_dutch_pinyin     # F5 Arrow dataset for finetune_cli.py

# 1. cut CGN recordings into 16 kHz utterance clips + metadata.csv (audio_file|text)
python Datasets/scripts/cgn/prep_cgn_f5.py --root "$ROOT" --out "$CLIPS"

# 2. tokenize + build the F5 Arrow dataset (train/valid + vocab)
python -m accent_vector.data_preprocess prepare \
    --metadata   "$CLIPS/metadata.csv" \
    --audio-root "$CLIPS" \
    --out-dir    "$DS" \
    --lora-label 0

echo "done: dataset at $DS"

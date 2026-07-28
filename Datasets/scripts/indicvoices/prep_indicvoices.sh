#!/bin/bash
# Eddie (SGE) batch job: build one IndicVoices-R accent dataset for F5-TTS. CPU-only.
# Runs once per accent -- set IVLANG on the qsub line:
#   qsub -v IVLANG=hi Datasets/scripts/indicvoices/prep_indicvoices.sh   # Hindi
#   qsub -v IVLANG=bn Datasets/scripts/indicvoices/prep_indicvoices.sh   # Bengali
#
#$ -N iv_prep
#$ -cwd
#$ -l h_rt=12:00:00
#$ -l h_vmem=32G
#$ -o logs/iv_prep.$JOB_ID.out
#$ -e logs/iv_prep.$JOB_ID.err

set -euo pipefail

: "${IVLANG:?set IVLANG=hi or IVLANG=bn on the qsub line: qsub -v IVLANG=hi ...}"

# --- environment (adjust to your Eddie setup) ---
. /etc/profile.d/modules.sh
module load anaconda
source activate f5-tts          # <-- your F5-TTS conda env (needs torchaudio, speechmos,
                                #     and the romaniser: ai4bharat-transliteration)

SCRATCH=/exports/eddie/scratch/s2247837/data
# IndicVoices-R on HuggingFace is per-language parquet folders (Hindi/, Bengali/) with
# audio embedded. Map the ISO IVLANG to that folder; override SRC=... if layout differs.
case "$IVLANG" in
  hi) LANGDIR=Hindi ;;
  bn) LANGDIR=Bengali ;;
  *)  LANGDIR=$IVLANG ;;
esac
SRC=${SRC:-$SCRATCH/indicvoices_r/$LANGDIR}   # downloaded IndicVoices-R parquet for this language

WORK=$SCRATCH/iv_${IVLANG}                 # selection + clips live here
CLIPS=$WORK/clips                        # wavs/ + metadata.csv
DS=$SCRATCH/iv_${IVLANG}_pinyin            # F5 Arrow dataset for finetune_cli.py

HOURS=${HOURS:-100}
CGN=Datasets/scripts/cgn                 # reuse the format-generic dnsmos/vocab_check
IV=Datasets/scripts/indicvoices
mkdir -p "$WORK"

# 1+2. balanced ~HOURS selection from read+extempore only (--no-conv-fallback: never
#    admit conversational) + decode only the selected clips' embedded audio -> wavs/ +
#    metadata.csv (NATIVE script text). Keeps native 48 kHz; F5's dataloader resamples
#    to 24k at load time (add --sr 24000 to pre-resample and halve on-disk size).
python $IV/prep_from_parquet.py \
    --parquet-dir "$SRC" --lang "$IVLANG" --hours "$HOURS" --out "$CLIPS" \
    --no-conv-fallback

# 3. DNSMOS quality filter (p808 >= 3.4) -> metadata.dnsmos.csv (audio-only; text unchanged)
python $CGN/dnsmos_filter.py --clips "$CLIPS" --min 3.4 --metric p808

# 4. romanise the survivors (native -> Latin) so F5's base vocab can tokenise them
python $IV/romanize.py --clips "$CLIPS" --lang "$IVLANG" --backend indicxlit \
    --in-name metadata.dnsmos.csv --out-name metadata.roman.csv

# 5. verify 0 OOV against the pretrained vocab BEFORE training
python $CGN/vocab_check.py --metadata "$CLIPS/metadata.roman.csv"

# 6. tokenize + build the F5 Arrow dataset
python -m accent_vector.data_preprocess prepare \
    --metadata   "$CLIPS/metadata.roman.csv" \
    --audio-root "$CLIPS" \
    --out-dir    "$DS" \
    --lora-label 0

echo "done: $IVLANG dataset at $DS"

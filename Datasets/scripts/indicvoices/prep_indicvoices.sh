#!/bin/bash
# Eddie (SGE) batch job: build one IndicVoices-R accent dataset for F5-TTS. CPU-only.
# Runs once per accent -- set IVLANG on the qsub line:
#   qsub -v IVLANG=hi Datasets/scripts/indicvoices/prep_indicvoices.sh   # Hindi
#   qsub -v IVLANG=bn Datasets/scripts/indicvoices/prep_indicvoices.sh   # Bengali
#
# RESUMABLE: each step is skipped if its output already exists, so re-submitting after
# a wall-time kill continues from where it stopped (e.g. decode done -> only DNSMOS runs).
# DNSMOS is the bottleneck (~10h single-threaded for ~70h of audio), so it runs in
# parallel across the requested cores; a 100h dataset needs the big h_rt below even so.
#
#$ -N iv_prep
#$ -cwd
#$ -pe sharedmem 8          # cores for parallel DNSMOS (JOBS below matches this)
#$ -l h_rt=48:00:00
#$ -l h_vmem=8G             # PER SLOT on sharedmem -> 8x8=64G total
#$ -o logs/iv_prep.$JOB_ID.out
#$ -e logs/iv_prep.$JOB_ID.err

set -euo pipefail

: "${IVLANG:?set IVLANG=hi or IVLANG=bn on the qsub line: qsub -v IVLANG=hi ...}"

# --- environment (adjust to your Eddie setup) ---
. /etc/profile.d/modules.sh
module load anaconda
source activate f5-tts          # F5-TTS conda env: needs torchaudio, pyarrow, soundfile,
                                #   speechmos (DNSMOS), and indic-transliteration (romaniser).
                                #   NOT ai4bharat-transliteration: it needs an online model
                                #   download + fairseq, neither of which works on a compute node.

SCRATCH=/exports/eddie/scratch/s2247837/data
# IndicVoices-R on HuggingFace is per-language parquet folders (Hindi/, Bengali/) with
# audio embedded. Map the ISO IVLANG to that folder; override SRC=... if layout differs.
case "$IVLANG" in
  hi) LANGDIR=Hindi ;;
  bn) LANGDIR=Bengali ;;
  *)  LANGDIR=$IVLANG ;;
esac
SRC=${SRC:-$SCRATCH/indicvoices_r/$LANGDIR}   # downloaded IndicVoices-R parquet for this language

WORK=$SCRATCH/iv_${IVLANG}
CLIPS=$WORK/clips                        # wavs/ + metadata.csv (+ .dnsmos.csv + .roman.csv)

HOURS=${HOURS:-100}
JOBS=${JOBS:-8}                          # DNSMOS worker processes; match -pe sharedmem above
CGN=Datasets/scripts/cgn                 # reuse the format-generic dnsmos/vocab_check
IV=Datasets/scripts/indicvoices
F5_ROOT=${F5_ROOT:-"$PWD/F5-TTS"}
mkdir -p "$WORK"

# locate the base vocab for vocab_check (Eddie checkouts vary: data/ may be absent)
VOCAB=${VOCAB:-}
if [ -z "$VOCAB" ]; then
  for c in "$F5_ROOT/data/vocab.txt" "$F5_ROOT/examples/vocab.txt" \
           "$F5_ROOT/data/dutch_pinyin/vocab.txt"; do
    [ -f "$c" ] && VOCAB="$c" && break
  done
fi

# 1+2. balanced ~HOURS selection from read+extempore only (--no-conv-fallback) + decode
#    only the selected clips' embedded audio -> wavs/ + metadata.csv (NATIVE script text).
#    Native 48 kHz; F5's dataloader resamples to 24k (add --sr 24000 to halve on-disk size).
if [ ! -f "$CLIPS/metadata.csv" ]; then
    python $IV/prep_from_parquet.py \
        --parquet-dir "$SRC" --lang "$IVLANG" --hours "$HOURS" --out "$CLIPS" \
        --no-conv-fallback
else
    echo "skip decode: $CLIPS/metadata.csv exists"
fi

# 3. DNSMOS quality filter (p808 >= 3.4) -> metadata.dnsmos.csv. Parallel across $JOBS cores.
if [ ! -f "$CLIPS/metadata.dnsmos.csv" ]; then
    python $CGN/dnsmos_filter.py --clips "$CLIPS" --min 3.4 --metric p808 --jobs "$JOBS"
else
    echo "skip DNSMOS: $CLIPS/metadata.dnsmos.csv exists"
fi

# 4. romanise survivors (native -> Latin) via offline rule-based sanscript (HK, ASCII).
#    ascii-fold (default) drops any residual non-ASCII so the output is vocab-safe.
if [ ! -f "$CLIPS/metadata.roman.csv" ]; then
    python $IV/romanize.py --clips "$CLIPS" --lang "$IVLANG" \
        --backend indic-translit --scheme HK \
        --in-name metadata.dnsmos.csv --out-name metadata.roman.csv
else
    echo "skip romanize: $CLIPS/metadata.roman.csv exists"
fi

# 5. verify 0 OOV against the pretrained vocab (the finetune's `prepare` will build the
#    Arrow from this CSV -- so this is the gate; no Arrow is built here on purpose).
python $CGN/vocab_check.py --metadata "$CLIPS/metadata.roman.csv" ${VOCAB:+--vocab "$VOCAB"}

echo "done: $IVLANG -> $CLIPS/metadata.roman.csv (feed this to the finetune as METADATA_CSV)"

#!/bin/bash
# Eddie (SGE) CPU ARRAY: score ONE alpha-sweep dir per task with the RQ eval suite --
# rq1 (accent_cs / speaker_sim / WER / LID) + rq3 (segmental/suprasegmental) + UTMOS.
# No GPU. Models are pre-staged by eddie_eval_setup.sh; jobs run offline.
# Submit via scripts/submit_eval_grid.sh (builds the manifest and runs qsub -t 1-N ...).
#
#$ -N eval_array
#$ -cwd
#$ -pe sharedmem 4           # CPU slots; total mem = h_vmem x NSLOTS
#$ -l h_rt=03:00:00          # one sweep (5 alphas x ~6 utts) incl. model reloads; generous
#$ -l h_vmem=8G              # per slot -> 32G total. CPU eval: no CUDA vmem blowup, this is plenty.
#$ -o logs/eval.$JOB_ID.$TASK_ID.out
#$ -e logs/eval.$JOB_ID.$TASK_ID.err
#$ -P ppls_slpgpu            # VERIFY: use a CPU-eligible project if the GPU project rejects CPU jobs
#$ -M s2247837@ed.ac.uk
#$ -m a                      # email on abort only
set -euo pipefail

. /etc/profile.d/modules.sh
module load anaconda

ACCENT_DIR="${SGE_O_WORKDIR:-$PWD}"          # submitted from AccentVector
REPO="$(cd "$ACCENT_DIR/.." && pwd)"
EVAL_DIR="$REPO/Evaluation"

# --- envs + offline caches (must match eddie_eval_setup.sh) ---
EVAL_ENV=${EVAL_ENV:-slp-eval}
GENAID_ENV=${GENAID_ENV:-genaid}
UTMOS_ENV=${UTMOS_ENV:-slp-utmos}
export HF_HOME=${HF_HOME:-/exports/eddie/scratch/$USER/hfcache}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}   # setup's warm pass overrides this to 0
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=${NSLOTS:-4}
# evaluation_functions.py shells out to the genaid env for cs_accent/speaker_sim/LID:
export GENAID_DIR="$EVAL_DIR/GenAID/recipes/CommonAccent"
export GENAID_PYTHON="$(conda run -n "$GENAID_ENV" which python)"
UTMOS_PYTHON="$(conda run -n "$UTMOS_ENV" which python)"

conda activate "$EVAL_ENV"
export PYTHONPATH="$ACCENT_DIR:$EVAL_DIR:${PYTHONPATH:-}"

MANIFEST="${1:?pass the task manifest (TSV) as arg 1}"
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 1; }
: "${SGE_TASK_ID:?submit as an array job (qsub -t 1-N) -- or set SGE_TASK_ID=1 to warm}"

# columns: 1 ACCENT 2 SPEAKER 3 REF_KIND 4 SWEEP_DIR 5 TRANSCRIPTS 6 REF_WAV 7 GT_DIR(optional)
LINE="$(sed -n "${SGE_TASK_ID}p" "$MANIFEST")"
[ -n "$LINE" ] || { echo "no manifest row for task $SGE_TASK_ID" >&2; exit 1; }
IFS=$'\t' read -r ACCENT SPEAKER REF_KIND SWEEP_DIR TRANSCRIPTS REF_WAV GT_DIR <<< "$LINE"

echo "[eval $JOB_ID.$SGE_TASK_ID] $ACCENT/$REF_KIND/$SPEAKER  $(basename "$SWEEP_DIR")"
echo "  gt=${GT_DIR:-<none: cs_accent+rq3 skipped>}  offline=$HF_HUB_OFFLINE"
for f in "$SWEEP_DIR" "$TRANSCRIPTS" "$REF_WAV"; do
  [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done

# --- RQ1 (+ leakage/LID). cs_accent only when a GT dir is present. ---
python -m accent_vector.experiments.rq1_reproduction \
  --sweep-dir "$SWEEP_DIR" --transcripts "$TRANSCRIPTS" --ref-wav "$REF_WAV" --lid \
  ${GT_DIR:+--accent-ref "$GT_DIR"} --out-csv "$SWEEP_DIR/rq1.csv"

# --- RQ3 decomposition: needs natural target-accent clips. ---
if [ -n "${GT_DIR:-}" ]; then
  python -m accent_vector.experiments.rq3_decomposition \
    --sweep-dir "$SWEEP_DIR" --natural-ref "$GT_DIR" --out-csv "$SWEEP_DIR/rq3.csv"
else
  echo "  [rq3] skipped (no GT_DIR for $ACCENT/$SPEAKER)"
fi

# --- UTMOS (own env, reference-free). ---
"$UTMOS_PYTHON" "$EVAL_DIR/score_utmos.py" --sweep-dir "$SWEEP_DIR" --out-csv "$SWEEP_DIR/utmos.csv"

echo "[eval $JOB_ID.$SGE_TASK_ID] done -> $SWEEP_DIR/{rq1,rq3,utmos}.csv"

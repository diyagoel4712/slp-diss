#!/bin/bash
# Eddie (SGE) ARRAY wrapper for the accent alpha-sweep -- one array TASK per
# (accent x ref_kind x speaker x transcript-shard) row of a manifest, so the whole
# 3-accent x 2-ref x 2-speaker inference fans out across as many GPUs as the queue
# grants (capped by qsub -tc). Submit via scripts/submit_infer_sweeps.sh, which builds
# the manifest and runs:
#   qsub -t 1-<N> -tc <MAX> -v ALPHAS="0 0.25 ..." scripts/eddie_infer_array.sh <manifest.tsv>
#
# Each task is an INDEPENDENT SGE job (its own GPU allocation + cgroup isolation), so
# CUDA_VISIBLE_DEVICES=0 inside infer_sweep.sh is correct here exactly as for a single job.
#
#$ -N infer_array
#$ -cwd
#$ -q gpu,gpu_new    # gpu queue was fully disabled 2026-08; gpu_new holds current GPU capacity. Both listed so it survives gpu being re-enabled.
#$ -l gpu=1
#$ -l h_rt=02:00:00          # one shard of one combo is short; generous ceiling
#$ -l h_vmem=64G             # VIRTUAL mem ceiling; CUDA reserves ~50G vmem/process (see [[eddie-hvmem-cuda-gotcha]])
#$ -o logs/infer.$JOB_ID.$TASK_ID.out
#$ -e logs/infer.$JOB_ID.$TASK_ID.err
#$ -P ppls_slpgpu
#$ -M s2247837@ed.ac.uk
#$ -m a                      # email on abort only (an array of successes shouldn't spam)

set -euo pipefail

. /etc/profile.d/modules.sh
module load anaconda
conda activate f5-tts
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

ACCENT_DIR="${SGE_O_WORKDIR:-$PWD}"
export F5_ROOT=${F5_ROOT:-"$ACCENT_DIR/../F5-TTS"}
export PRETRAIN=${PRETRAIN:-/exports/eddie/scratch/s2247837/ckpts/F5TTS_v1_Base/model_1250000.safetensors}

MANIFEST="${1:?pass the task manifest (TSV) as the first argument}"
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 1; }
: "${SGE_TASK_ID:?this script must be submitted as an array job (qsub -t 1-N)}"

# --- pick this task's row. Columns 1-8 required; 9-11 optional (empty in the plain
#     final-vector manifest, set by the checkpoint-grid submitter):
#   1 ACCENT 2 SPEAKER 3 REF_KIND 4 REF_AUDIO 5 REF_TEXT 6 RUN_DIR 7 SHARD_INDEX 8 SHARD_COUNT
#   9 VECTOR(per-row, e.g. a checkpoint) 10 TRANSCRIPTS(per-row) 11 OUT_SUBDIR(e.g. step_NNN) ---
LINE="$(sed -n "${SGE_TASK_ID}p" "$MANIFEST")"
[ -n "$LINE" ] || { echo "no manifest row for task $SGE_TASK_ID in $MANIFEST" >&2; exit 1; }
IFS=$'\t' read -r ACCENT SPEAKER REF_KIND REF_AUDIO REF_TEXT RUN_DIR SHARD_INDEX SHARD_COUNT \
    ROW_VECTOR ROW_TRANSCRIPTS OUT_SUBDIR <<< "$LINE"

export ACCENT_NAME="$ACCENT"
export SPEAKER REF_KIND REF_AUDIO RUN_DIR SHARD_INDEX SHARD_COUNT
export CONFIG="$RUN_DIR/config.yaml"
export VOCAB="$RUN_DIR/vocab.txt"
export PYTHONPATH="$F5_ROOT/src:$ACCENT_DIR:${PYTHONPATH:-}"

# per-row VECTOR (col 9) overrides the accent's final vector -- e.g. a training checkpoint.
# A full model_<step>.pt is sliced to its LoRA vector on the fly; a lora_<step>.pt snapshot
# (or any explicit vector) is used as-is.
if [ -n "${ROW_VECTOR:-}" ]; then
  case "$(basename "$ROW_VECTOR")" in
    model_*.pt)
      TMPVEC="$(mktemp -d)/lora_$(basename "$ROW_VECTOR")"
      echo "[array] slicing LoRA vector from $ROW_VECTOR"
      python -m accent_vector.extract_vector extract-lora --checkpoint "$ROW_VECTOR" --out "$TMPVEC" --source model
      export VECTOR="$TMPVEC" ;;
    *) export VECTOR="$ROW_VECTOR" ;;
  esac
else
  export VECTOR=${VECTOR:-"$ACCENT_DIR/vectors/${ACCENT}.pt"}
fi

# per-row TRANSCRIPTS (col 10) overrides infer_sweep.sh's default (e.g. per-speaker eval set).
if [ -n "${ROW_TRANSCRIPTS:-}" ]; then
  case "$ROW_TRANSCRIPTS" in /*) export TRANSCRIPTS="$ROW_TRANSCRIPTS";; *) export TRANSCRIPTS="$ACCENT_DIR/$ROW_TRANSCRIPTS";; esac
fi

# ALPHAS arrives via qsub -v (space-separated to dodge the comma-splitting); normalise here.
export ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1.0}"
export ALPHAS="${ALPHAS// /,}"
# OUT_DIR: results/<accent>[/<results_tag>]/<ref_kind>/<speaker>[/<out_subdir, e.g. step_NNN>].
# RESULTS_TAG (optional, via qsub -v) separates otherwise-identical sweeps whose paths
# would otherwise clash -- e.g. a hyperparameter cell (lr3e5_r16) whose step_<step> dirs
# collide in the shared results/<accent> tree. Base-accent asset lookups are unaffected.
export OUT_DIR=${OUT_DIR:-"$ACCENT_DIR/results/${ACCENT}${RESULTS_TAG:+/$RESULTS_TAG}/${REF_KIND}${SPEAKER:+/$SPEAKER}${OUT_SUBDIR:+/$OUT_SUBDIR}"}

# REF_TEXT may be a literal or a file path (romanised L1 transcript file for hi/bn); cat if a file.
if [ -f "$REF_TEXT" ]; then REF_TEXT="$(cat "$REF_TEXT")"
elif [ -f "$ACCENT_DIR/$REF_TEXT" ]; then REF_TEXT="$(cat "$ACCENT_DIR/$REF_TEXT")"; fi
export REF_TEXT

echo "[array $JOB_ID.$SGE_TASK_ID] accent=$ACCENT spk=$SPEAKER kind=$REF_KIND shard=$SHARD_INDEX/$SHARD_COUNT"
echo "  vector=$VECTOR  alphas=$ALPHAS  out=$OUT_DIR"
nvidia-smi -L || true

for f in "$VECTOR" "$CONFIG" "$VOCAB" "$REF_AUDIO"; do
    [ -f "$f" ] || { echo "ERROR: task $SGE_TASK_ID missing required file: $f" >&2; exit 1; }
done

# per-task provenance dir (unique -> no cross-task clobber when shards share an OUT_DIR).
bash "$ACCENT_DIR/scripts/record_provenance.sh" "$ACCENT_DIR" "$F5_ROOT" \
    "$OUT_DIR/provenance/task_${JOB_ID}_${SGE_TASK_ID}" \
    || echo "warning: provenance capture failed (continuing)"

bash "$ACCENT_DIR/scripts/infer_sweep.sh"

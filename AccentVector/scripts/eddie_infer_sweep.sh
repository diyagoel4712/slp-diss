#!/bin/bash
# Eddie (SGE) GPU wrapper for scripts/infer_sweep.sh -- the alpha sweep for ONE accent.
#   cd /exports/chss/eddie/ppls/groups/slpgpustorage/users/s2247837/slp-diss/AccentVector && mkdir -p logs
#   qsub scripts/eddie_infer_sweep.sh
# Override any env var at submit time, e.g.:
#   qsub -N infer_dutch -v ACCENT_NAME=dutch,RUN_DIR=/exports/eddie/scratch/s2247837/accentvector-exps/F5TTS_v1_LoRA_dutch/2026-07-24_00-34-07 \
#        scripts/eddie_infer_sweep.sh
# The job name is static (SGE parses -N before the script runs); override it on the
# command line to match, e.g. -N infer_dutch.
#
#$ -N infer_sweep
#$ -cwd
#$ -q gpu
#$ -l gpu=1
#$ -l h_rt=04:00:00          # a 6-alpha sweep over the eval set is ~10-20 min; generous ceiling
#$ -l h_vmem=64G             # VIRTUAL mem ceiling; CUDA reserves ~50G vmem/process. Keep generous.
#$ -o logs/infer.$JOB_ID.out
#$ -e logs/infer.$JOB_ID.err
#$ -P ppls_slpgpu
#$ -M s2247837@ed.ac.uk
#$ -m eab                    # email on end (e) and abort (a)

set -euo pipefail

# --- environment (mirror the finetune wrapper) ---
# No `module load cuda`: the f5-tts torch wheel bundles its own CUDA runtime.
. /etc/profile.d/modules.sh
module load anaconda
conda activate f5-tts

# no internet on the node: the vocoder is loaded locally (is_local=True in the run's
# config.yaml), so never let a stray HF call hang the GPU -- fail fast.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

# SGE runs a spooled COPY of this script, so $0 is not the scripts/ path. With -cwd the
# submission dir (AccentVector) is SGE_O_WORKDIR; fall back to $PWD outside SGE.
ACCENT_DIR="${SGE_O_WORKDIR:-$PWD}"

# --- run config (all overridable via `qsub -v KEY=VAL,...`) ---
export F5_ROOT=${F5_ROOT:-"$ACCENT_DIR/../F5-TTS"}
export ACCENT_NAME=${ACCENT_NAME:-dutch}
# training run dir holds config.yaml + vocab.txt (needed to rebuild the base+LoRA model).
export RUN_DIR=${RUN_DIR:-/exports/eddie/scratch/s2247837/accentvector-exps/F5TTS_v1_LoRA_dutch/2026-07-24_00-34-07}
export CONFIG=${CONFIG:-"$RUN_DIR/config.yaml"}
export VOCAB=${VOCAB:-"$RUN_DIR/vocab.txt"}
export VECTOR=${VECTOR:-"$ACCENT_DIR/vectors/${ACCENT_NAME}.pt"}
export PRETRAIN=${PRETRAIN:-/exports/eddie/scratch/s2247837/ckpts/F5TTS_v1_Base/model_1250000.safetensors}
export ALPHAS=${ALPHAS:-"0,0.2,0.4,0.6,0.8,1.0"}
# qsub -v uses commas to separate variables, so a comma-separated ALPHAS value collapses
# to its first element (ALPHAS="0,0.25,..." -> ALPHAS=0). Pass ALPHAS space-separated on
# the qsub line (e.g. ALPHAS="0 0.25 0.5 0.75 1.0") and normalise spaces to commas here.
export ALPHAS="${ALPHAS// /,}"
# reference condition -- WHICH clip F5 clones. This is the accent-decoupling control:
#   l1     (default) the accent's native-language reference. Paper-faithful, but accent
#                    comes from BOTH cloning the L1 clip AND the vector -> the confounded
#                    baseline. REF_AUDIO/REF_TEXT REQUIRED (must match the clip exactly).
#   native           a neutral native-English clip whose accent is NOT the target, so at
#                    alpha=0 the output is neutral English and any target-accent signal
#                    that emerges as alpha climbs is attributable to the VECTOR alone.
#                    Defaults to refs/native_ga.{wav,txt} (supply the clip) unless overridden.
# REF_KIND (a) picks that default for `native` and (b) suffixes OUT_DIR so the two
# conditions land in sibling trees results/<accent>/<ref_kind>/ for a clean comparison.
export REF_KIND=${REF_KIND:-l1}
case "$REF_KIND" in
    native)
        export REF_AUDIO=${REF_AUDIO:-"$ACCENT_DIR/refs/native_ga.wav"}
        export REF_TEXT=${REF_TEXT:-"$ACCENT_DIR/refs/native_ga.txt"}
        ;;
    l1)
        # no sensible default for an L1 clip, so fail loudly if unset.
        export REF_AUDIO=${REF_AUDIO:?set REF_AUDIO=refs/<accent>_l1.wav (the fixed L1 reference clip)}
        export REF_TEXT=${REF_TEXT:?set REF_TEXT to the exact transcript of REF_AUDIO (or a path to a file holding it)}
        ;;
    *)
        echo "ERROR: REF_KIND must be 'l1' or 'native' (got '$REF_KIND')" >&2; exit 1
        ;;
esac
# REF_TEXT may be either the literal transcript or a path to a file containing it
# (handy for long L1 references -- avoids retyping/mis-quoting in the qsub -v line).
# Resolve relative paths against the submission dir, mirroring the other inputs.
if [ -f "$REF_TEXT" ]; then
    REF_TEXT="$(cat "$REF_TEXT")"
elif [ -f "$ACCENT_DIR/$REF_TEXT" ]; then
    REF_TEXT="$(cat "$ACCENT_DIR/$REF_TEXT")"
fi
export REF_TEXT
export OUT_DIR=${OUT_DIR:-"$ACCENT_DIR/results/${ACCENT_NAME}/${REF_KIND}"}
# LoRA is the paper-matching default in infer_sweep.sh; set LORA=0 for a merged sweep.
export LORA=${LORA:-1}

echo "accent=$ACCENT_NAME  vector=$VECTOR"
echo "config=$CONFIG  vocab=$VOCAB"
echo "ref_kind=$REF_KIND  ref=$REF_AUDIO  alphas=$ALPHAS  out=$OUT_DIR"
nvidia-smi -L || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'avail', torch.cuda.is_available())"

# fail early with a clear message if the vector/config/vocab aren't where we expect.
for f in "$VECTOR" "$CONFIG" "$VOCAB" "$REF_AUDIO"; do
    [ -f "$f" ] || { echo "ERROR: missing required file: $f" >&2; exit 1; }
done

# record run provenance (best-effort; must never kill the GPU job).
bash "$ACCENT_DIR/scripts/record_provenance.sh" "$ACCENT_DIR" "$F5_ROOT" "$OUT_DIR/provenance" \
    || echo "warning: provenance capture failed (continuing)"

# the sweep itself (env vars above are read by infer_sweep.sh).
bash "$ACCENT_DIR/scripts/infer_sweep.sh"

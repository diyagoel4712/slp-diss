#!/bin/bash
# Dutch CHECKPOINT x alpha inference grid on Eddie (feeds RQ2-behavioural + RQ1).
#
# Grid = checkpoints every STEP_INTERVAL steps (default 5k; up to LAST_STEP=73400), snapped to
#        nearest snapshot so runs of different length align -- STEP_INTERVAL=0 falls back to
#        N_CHECKPOINTS evenly-spaced per-run fractions
#        x REF_KINDS {l1 native} x SPEAKERS {m f} x transcript-SHARDS,
# each an alpha sweep {0 0.25 0.5 0.75 1.0} ->
#   results/dutch/<ref_kind>/<speaker>/step_<step>/alpha_<a>/utt####.wav
# so rq1_reproduction (per step) + rq2_behavioural compare matched-alpha across training.
#
# Dutch assets (all local in the repo): L1 = prompts/dutch/dutch_{m,f}.{wav,_ref.txt};
# GAE = prompts/GAE/gae_{m,f}.{wav,txt}; per-speaker sentences = transcripts/dutch/dutch_{m,f}_eval.txt.
# Checkpoints are discovered in RUN_DIR/ckpts (prefers lora_<step>.pt snapshots, else
# model_<step>.pt which the array task slices on the fly).
#
# Runs on a LOGIN node (globs the Eddie ckpts dir), builds a manifest, submits ONE array job.
#   bash scripts/submit_dutch_ckpt_grid.sh
#   DRY_RUN=1 bash scripts/submit_dutch_ckpt_grid.sh
#   STEP_INTERVAL=5000 SHARDS=2 MAX_CONCURRENT=12 bash scripts/submit_dutch_ckpt_grid.sh
#   STEP_INTERVAL=0 N_CHECKPOINTS=8 bash scripts/submit_dutch_ckpt_grid.sh   (even-spacing fallback)
set -uo pipefail

ACCENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ACCENT_DIR"; mkdir -p logs
ACCENT=dutch

RUN_DIR=${RUN_DIR:-/exports/eddie/scratch/s2247837/accentvector-exps/F5TTS_v1_LoRA_dutch/2026-07-24_00-34-07}
CKPT_DIR=${CKPT_DIR:-"$RUN_DIR/ckpts/snapshots"}
LAST_STEP=${LAST_STEP:-73400}
STEP_INTERVAL=${STEP_INTERVAL:-5000}  # pick checkpoints on a FIXED step grid (5k,10k,...) up to
                                      # LAST_STEP, snapped to nearest available -> runs of slightly
                                      # different length align on the same x-axis for overlay plots.
                                      # 0 = fall back to N_CHECKPOINTS evenly-spaced per-run fractions.
N_CHECKPOINTS=${N_CHECKPOINTS:-8}  # only used when STEP_INTERVAL=0
REF_KINDS=${REF_KINDS:-"l1 native"}
SPEAKERS=${SPEAKERS:-"m f"}
ALPHAS=${ALPHAS:-"0 0.25 0.5 0.75 1.0"}     # space-sep; passed via -v, wrapper -> commas
SHARDS=${SHARDS:-1}
MAX_CONCURRENT=${MAX_CONCURRENT:-8}
NATIVE_PREFIX=${NATIVE_PREFIX:-prompts/GAE/gae}
# Optional label separating this sweep -> results/<accent>/<tag>/<ref_kind>/... so cells
# whose step_<step> dirs would otherwise clash (e.g. a hparam grid: RESULTS_TAG=lr3e5_r16)
# stay distinct. Must be space/comma-free (rides qsub -v). Empty = old flat layout.
RESULTS_TAG=${RESULTS_TAG:-}

# --- run dir sanity ---
for f in "$RUN_DIR/config.yaml" "$RUN_DIR/vocab.txt"; do
  [ -f "$f" ] || { echo "run dir incomplete: missing $f" >&2; exit 1; }
done

# --- discover checkpoints: prefer lora_<step>.pt snapshots, else model_<step>.pt ---
discover() { ls "$CKPT_DIR"/$1_*.pt 2>/dev/null | grep -vE 'model_last|model_diff' \
    | sed -E "s#.*/$1_([0-9]+)\.pt#\1#" | sort -n; }
CKPT_PREFIX=lora; STEPS=()
while IFS= read -r x; do [ -n "$x" ] && STEPS+=("$x"); done < <(discover lora)
if [ "${#STEPS[@]}" -eq 0 ]; then
  CKPT_PREFIX=model
  while IFS= read -r x; do [ -n "$x" ] && STEPS+=("$x"); done < <(discover model)
fi
[ "${#STEPS[@]}" -gt 0 ] || { echo "no lora_<step>.pt or model_<step>.pt in $CKPT_DIR" >&2; exit 1; }

# Select checkpoints to sweep. Default: a FIXED step grid (STEP_INTERVAL: 5k,10k,...) up to
# LAST_STEP, each snapped to the nearest available snapshot -- so runs of slightly different
# length are evaluated at the SAME steps and overlay cleanly. STEP_INTERVAL=0 -> N_CHECKPOINTS
# evenly-spaced fractions of this run's own length.
SELECTED=$(printf '%s\n' "${STEPS[@]}" | python3 -c '
import sys
last=int(sys.argv[1]); n=int(sys.argv[2]); interval=int(sys.argv[3])
avail=sorted(set(int(x) for x in sys.stdin.read().split()))
cands=[s for s in avail if s<=last] or avail
if interval>0:
    targets=list(range(interval, last+1, interval)) or [last]
else:
    targets=[round(k*last/n) for k in range(1,n+1)]
picked=[]
for t in targets:
    s=min(cands, key=lambda a: abs(a-t))
    if s not in picked: picked.append(s)
print(" ".join(str(s) for s in sorted(picked)))
' "$LAST_STEP" "$N_CHECKPOINTS" "$STEP_INTERVAL")
sel_desc=$([ "$STEP_INTERVAL" -gt 0 ] && echo "every ${STEP_INTERVAL} steps" || echo "${N_CHECKPOINTS} evenly-spaced")
echo "checkpoints: ${#STEPS[@]} ${CKPT_PREFIX}_<step>.pt available; picked $(echo "$SELECTED" | wc -w | tr -d ' ') ($sel_desc) up to $LAST_STEP: $SELECTED"

MANIFEST="logs/dutch_ckpt_tasks.$(date +%Y%m%d_%H%M%S).tsv"; : > "$MANIFEST"
n_combos=0 n_skipped=0
chk() { [ -e "$1" ] && return 0; printf '    \033[31mmissing\033[0m %s\n' "$1"; return 1; }

for step in $SELECTED; do
  vec="$CKPT_DIR/${CKPT_PREFIX}_${step}.pt"
  for kind in $REF_KINDS; do for spk in $SPEAKERS; do
    name="$kind/$spk/step_$step"
    local_miss=0
    if [ "$kind" = l1 ]; then
      ra="prompts/dutch/dutch_${spk}.wav"; rt="prompts/dutch/dutch_${spk}_ref.txt"
    else
      ra="${NATIVE_PREFIX}_${spk}.wav"; rt="${NATIVE_PREFIX}_${spk}.txt"
    fi
    tx="transcripts/dutch/dutch_${spk}_eval.txt"
    chk "$ra" || local_miss=1
    chk "$rt" || local_miss=1
    chk "$tx" || local_miss=1
    if [ "$local_miss" = 1 ]; then printf '  \033[31mSKIP\033[0m %s (assets above)\n' "$name"; n_skipped=$((n_skipped+1)); continue; fi
    for ((s=0; s<SHARDS; s++)); do
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$ACCENT" "$spk" "$kind" "$ra" "$rt" "$RUN_DIR" "$s" "$SHARDS" "$vec" "$tx" "step_$step" >> "$MANIFEST"
    done
    n_combos=$((n_combos+1))
  done; done
done

N=$(wc -l < "$MANIFEST" | tr -d ' ')
echo "manifest: $MANIFEST  ($n_combos combos x $SHARDS shard(s) = $N tasks; $n_skipped skipped)"
[ "$N" -gt 0 ] || { echo "no runnable tasks (assets missing); prepare them and re-run." >&2; rm -f "$MANIFEST"; exit 1; }

QSUB=(qsub -t "1-$N" -tc "$MAX_CONCURRENT" -v "ALPHAS=$ALPHAS")
[ -n "$RESULTS_TAG" ] && QSUB+=(-v "RESULTS_TAG=$RESULTS_TAG")
QSUB+=(scripts/eddie_infer_array.sh "$ACCENT_DIR/$MANIFEST")
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "--- manifest rows ---"; cat "$MANIFEST"
  echo "--- would submit ---"; printf '  %q ' "${QSUB[@]}"; printf '\n'
else
  "${QSUB[@]}"
  echo "submitted array of $N tasks (-tc $MAX_CONCURRENT). qstat to watch; qdel <jobid> to cancel all."
fi
[ "$n_skipped" -eq 0 ] || echo "note: $n_skipped combo(s) skipped for missing assets (see above)." >&2

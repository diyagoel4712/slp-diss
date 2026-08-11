#!/bin/bash
# Build the eval manifest over EXISTING sweep dirs and submit the CPU array
# (scripts/eddie_eval_array.sh). One task per
# results/<accent>[/<tag>]/<ref_kind>/<speaker>/audio/step_*/ ; the array writes the
# CSVs to the sibling metrics/ tree (.../metrics/step_*/).
#
#   bash scripts/submit_eval_grid.sh                       # full grid, all accents/steps
#   STEP_SCOPE=final bash scripts/submit_eval_grid.sh      # only the final checkpoint per cell
#   ACCENTS="dutch" REF_KINDS="native" bash scripts/submit_eval_grid.sh
#   DRY_RUN=1 bash scripts/submit_eval_grid.sh             # print manifest + qsub line, don't submit
set -uo pipefail

ACCENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ACCENT_DIR"; mkdir -p logs
ACCENTS=${ACCENTS:-"dutch hindi bengali arabic"}
REF_KINDS=${REF_KINDS:-"l1 native"}
SPEAKERS=${SPEAKERS:-"m f"}
STEP_SCOPE=${STEP_SCOPE:-all}          # all | final
MAX_CONCURRENT=${MAX_CONCURRENT:-16}
# Must match the RESULTS_TAG the sweep was submitted with -> globs results/<accent>/<tag>/...
# Empty = old flat layout results/<accent>/... (untagged sweeps).
RESULTS_TAG=${RESULTS_TAG:-}

# per-accent, per-speaker L1 reference basename (matches submit_indic_ckpt_grid.sh)
l1base() { case "$1/$2" in
  dutch/m)   echo prompts/dutch/dutch_m;;   dutch/f)   echo prompts/dutch/dutch_f;;
  hindi/m)   echo prompts/hindi/hi_M_04;;   hindi/f)   echo prompts/hindi/hi_F_02;;
  bengali/m) echo prompts/bengali/bn_M_01;; bengali/f) echo prompts/bengali/bn_F_02;;
  # Arabic: held-out GlobalPhone speakers (must match l1base in submit_indic_ckpt_grid.sh).
  arabic/m)  echo prompts/arabic/ar_M_AR010;;  arabic/f)  echo prompts/arabic/ar_F_AR002;;
  # Mandarin: held-out FLEURS speakers (must match l1base in submit_indic_ckpt_grid.sh).
  mandarin/m) echo prompts/mandarin/mandarin_M_824;; mandarin/f) echo prompts/mandarin/mandarin_F_369;;
esac; }
gdir() { [ "$1" = f ] && echo female || echo male; }   # speaker -> GT gender dir

MANIFEST="logs/eval_tasks.$(date +%Y%m%d_%H%M%S).tsv"; : > "$MANIFEST"
n=0 n_nogt=0
for a in $ACCENTS; do for r in $REF_KINDS; do for s in $SPEAKERS; do
  base="$(l1base "$a" "$s")"
  [ "$r" = l1 ] && ref="$base.wav" || ref="prompts/GAE/gae_${s}.wav"
  tx="transcripts/$a/${a}_${s}_eval.txt"
  gt="ground_truth_refs/$a/$(gdir "$s")"; [ -d "$gt" ] || gt=""    # empty => cs_accent/rq3 skipped

  rroot="results/$a${RESULTS_TAG:+/$RESULTS_TAG}/$r/$s/audio"
  steps=$(ls -d "$rroot"/step_* 2>/dev/null | sort -t_ -k2 -n)
  [ -n "$steps" ] || { echo "  no sweeps under $rroot"; continue; }
  [ "$STEP_SCOPE" = final ] && steps=$(echo "$steps" | tail -1)

  for d in $steps; do
    [ -f "$tx" ]  || { printf '  \033[31mmissing\033[0m %s\n' "$tx";  continue; }
    [ -f "$ref" ] || { printf '  \033[31mmissing\033[0m %s\n' "$ref"; continue; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$a" "$s" "$r" \
      "$ACCENT_DIR/$d" "$ACCENT_DIR/$tx" "$ACCENT_DIR/$ref" "${gt:+$ACCENT_DIR/$gt}" >> "$MANIFEST"
    n=$((n+1)); [ -z "$gt" ] && n_nogt=$((n_nogt+1))
  done
done; done; done

N=$(wc -l < "$MANIFEST" | tr -d ' ')
echo "manifest: $MANIFEST  ($N tasks; $n_nogt with no GT dir -> cs_accent+rq3 skipped there)"
[ "$N" -gt 0 ] || { echo "no runnable sweeps found." >&2; rm -f "$MANIFEST"; exit 1; }

QSUB=(qsub -t "1-$N" -tc "$MAX_CONCURRENT" scripts/eddie_eval_array.sh "$ACCENT_DIR/$MANIFEST")
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "--- manifest ---"; cat "$MANIFEST"
  echo "--- would submit ---"; printf '  %q ' "${QSUB[@]}"; printf '\n'
else
  "${QSUB[@]}"
  echo "submitted $N tasks (-tc $MAX_CONCURRENT). qstat to watch; qdel <jobid> to cancel."
fi

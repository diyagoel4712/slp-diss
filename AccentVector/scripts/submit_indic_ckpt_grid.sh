#!/bin/bash
# Decoupled-accent (Hindi / Bengali / Arabic) CHECKPOINT x alpha inference grid on Eddie
# (mirrors submit_dutch_ckpt_grid.sh). All use FLEURS L1 prompts + SAA ground truth.
#
# Grid = N_CHECKPOINTS evenly-spaced training steps (up to LAST_STEP; 0 = auto = max available)
#        x REF_KINDS {l1 native} x SPEAKERS {m f} x transcript-SHARDS,
# each an alpha sweep {0 0.25 0.5 0.75 1.0} ->
#   results/<accent>/<ref_kind>/<speaker>/step_<step>/alpha_<a>/utt####.wav
#
# L1 refs = the held-out FLEURS prompts, ref text ROMANISED (indic-translit/HK, matching
# training) in *_ref.txt:
#   hindi:   m=prompts/hindi/hi_M_04     f=prompts/hindi/hi_F_02
#   bengali: m=prompts/bengali/bn_M_01   f=prompts/bengali/bn_F_02
# native (decoupling control) = prompts/GAE/gae_{m,f} (English). Eval transcripts = Stella
#   (transcripts/<accent>/<accent>_{m,f}_eval.txt), matching the SAA ground truth.
#
# Point RUN_DIR at the accent's training run dir (has config.yaml, vocab.txt, ckpts/snapshots):
#   ACCENT=hindi   RUN_DIR=/exports/.../F5TTS_v1_LoRA_hindi/<ts>   bash scripts/submit_indic_ckpt_grid.sh
#   ACCENT=bengali RUN_DIR=/exports/.../F5TTS_v1_LoRA_bengali/<ts> bash scripts/submit_indic_ckpt_grid.sh
#   ACCENT=arabic  RUN_DIR=/exports/.../F5TTS_v1_LoRA_arabic/<ts>  bash scripts/submit_indic_ckpt_grid.sh
#   DRY_RUN=1 ...   (print the manifest, don't submit)
#   N_CHECKPOINTS=8 SHARDS=2 MAX_CONCURRENT=12 ...
set -uo pipefail

ACCENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ACCENT_DIR"; mkdir -p logs
ACCENT=${ACCENT:?set ACCENT=hindi, bengali or arabic}

# Decoupled non-Latin accents (FLEURS L1 prompts + SAA GT), same grid pattern.
case "$ACCENT" in hindi|bengali|arabic) ;; *) echo "ACCENT must be hindi, bengali or arabic (got '$ACCENT')" >&2; exit 1;; esac
# L1 (native-language) reference basename for a given speaker (a function, not an assoc
# array, so it runs on old bash too). Empty output => unknown; caller treats as missing.
l1base() {
  case "$ACCENT/$1" in
    hindi/m)   echo prompts/hindi/hi_M_04;;
    hindi/f)   echo prompts/hindi/hi_F_02;;
    bengali/m) echo prompts/bengali/bn_M_01;;
    bengali/f) echo prompts/bengali/bn_F_02;;
    # Arabic: set to the actual FLEURS ar_{M,F}_NN IDs chosen at prompt-prep time.
    arabic/m)  echo prompts/arabic/ar_M_01;;
    arabic/f)  echo prompts/arabic/ar_F_01;;
  esac
}

RUN_DIR=${RUN_DIR:?set RUN_DIR=<the ${ACCENT} training run dir with config.yaml/vocab.txt/ckpts>}
CKPT_DIR=${CKPT_DIR:-"$RUN_DIR/ckpts/snapshots"}
LAST_STEP=${LAST_STEP:-0}          # 0 = auto: use the max available step
N_CHECKPOINTS=${N_CHECKPOINTS:-8}
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

# LAST_STEP=0 -> use the largest available step
[ "$LAST_STEP" -gt 0 ] || LAST_STEP=$(printf '%s\n' "${STEPS[@]}" | sort -n | tail -1)

# pick N steps evenly spaced by step value, ending at LAST_STEP, snapped to nearest available.
SELECTED=$(printf '%s\n' "${STEPS[@]}" | python3 -c '
import sys
last=int(sys.argv[1]); n=int(sys.argv[2])
avail=sorted(set(int(x) for x in sys.stdin.read().split()))
cands=[s for s in avail if s<=last] or avail
targets=[round(k*last/n) for k in range(1,n+1)]
picked=[]
for t in targets:
    s=min(cands, key=lambda a: abs(a-t))
    if s not in picked: picked.append(s)
print(" ".join(str(s) for s in sorted(picked)))
' "$LAST_STEP" "$N_CHECKPOINTS")
echo "[$ACCENT] checkpoints: ${#STEPS[@]} ${CKPT_PREFIX}_<step>.pt available; picked $(echo "$SELECTED" | wc -w | tr -d ' ') up to $LAST_STEP: $SELECTED"

MANIFEST="logs/${ACCENT}_ckpt_tasks.$(date +%Y%m%d_%H%M%S).tsv"; : > "$MANIFEST"
n_combos=0 n_skipped=0
chk() { [ -e "$1" ] && return 0; printf '    \033[31mmissing\033[0m %s\n' "$1"; return 1; }

for step in $SELECTED; do
  vec="$CKPT_DIR/${CKPT_PREFIX}_${step}.pt"
  for kind in $REF_KINDS; do for spk in $SPEAKERS; do
    name="$kind/$spk/step_$step"
    local_miss=0
    if [ "$kind" = l1 ]; then
      base="$(l1base "$spk")"; ra="${base}.wav"; rt="${base}_ref.txt"
    else
      ra="${NATIVE_PREFIX}_${spk}.wav"; rt="${NATIVE_PREFIX}_${spk}.txt"
    fi
    tx="transcripts/$ACCENT/${ACCENT}_${spk}_eval.txt"
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

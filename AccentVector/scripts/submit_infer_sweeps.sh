#!/bin/bash
# Submit ALL accent inference sweeps to Eddie as ONE SGE ARRAY job (multi-GPU).
#
# Tasks = accents {hindi bengali dutch} x REF_KINDS {l1 native} x SPEAKERS {m f}
#         x transcript-SHARDS  -> N array tasks, each an independent 1-GPU job,
#         packed onto free GPUs by SGE (capped by qsub -tc = MAX_CONCURRENT).
#   l1     clone the accent's native-LANGUAGE clip (paper-faithful baseline)
#   native clone a neutral General-American ENGLISH clip (accent = the VECTOR alone)
# Each task -> results/<accent>/<ref_kind>/<speaker>/alpha_<a>/utt####.wav. SHARDS>1
# splits the transcript list across sibling tasks that reassemble into one alpha_<a>/
# (disjoint global utt#### names), so a big transcript set fans out across GPUs while
# each task still builds the model ONCE (build-once is preserved; only the sweep splits).
#
# Runs on a LOGIN node: builds a manifest (logs/infer_tasks.<ts>.tsv), then
#   qsub -t 1-N -tc <MAX_CONCURRENT> -v ALPHAS="..." scripts/eddie_infer_array.sh <manifest>
# One submission; one `qdel <jobid>` cancels the whole grid.
#
#   bash scripts/submit_infer_sweeps.sh
#   DRY_RUN=1 bash scripts/submit_infer_sweeps.sh            # build+print manifest & qsub line, submit nothing
#   SHARDS=4 MAX_CONCURRENT=12 bash scripts/...              # 4-way transcript split, up to 12 GPUs at once
#   ACCENTS="dutch" REF_KINDS="native" SPEAKERS="f" bash ... # a subset
#   HINDI_RUN_DIR=... BENGALI_RUN_DIR=... bash scripts/...   # point at your finetune run dirs
#
# A local pre-flight OMITS (doesn't enqueue) any combo whose assets are missing.
set -uo pipefail

ACCENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ACCENT_DIR"
mkdir -p logs

# --- per-accent training run dir (holds config.yaml + vocab.txt). EDIT to your timestamps. ---
SCRATCH=${SCRATCH:-/exports/eddie/scratch/s2247837/accentvector-exps}
HINDI_RUN_DIR=${HINDI_RUN_DIR:-$SCRATCH/F5TTS_v1_LoRA_hindi/RUN_TIMESTAMP}
BENGALI_RUN_DIR=${BENGALI_RUN_DIR:-$SCRATCH/F5TTS_v1_LoRA_bengali/RUN_TIMESTAMP}
DUTCH_RUN_DIR=${DUTCH_RUN_DIR:-$SCRATCH/F5TTS_v1_LoRA_dutch/2026-07-24_00-34-07}

ACCENTS=${ACCENTS:-"hindi bengali dutch"}
REF_KINDS=${REF_KINDS:-"l1 native"}
SPEAKERS=${SPEAKERS:-"m f"}                     # one set of tasks per speaker
ALPHAS=${ALPHAS:-"0 0.25 0.5 0.75 1.0"}         # space-sep; passed via -v, wrapper -> commas
SHARDS=${SHARDS:-1}                             # transcript shards per combo (>1 fans a combo across GPUs)
MAX_CONCURRENT=${MAX_CONCURRENT:-8}             # qsub -tc: max array tasks running at once (~ GPUs used)
NATIVE_PREFIX=${NATIVE_PREFIX:-refs/native_ga}  # neutral GA English clips: <prefix>_m.wav / _f.wav (+ .txt)

run_dir_for() { case "$1" in
    hindi)   echo "$HINDI_RUN_DIR" ;;
    bengali) echo "$BENGALI_RUN_DIR" ;;
    dutch)   echo "$DUTCH_RUN_DIR" ;;
    *)       echo "" ;;
  esac; }

MANIFEST="logs/infer_tasks.$(date +%Y%m%d_%H%M%S).tsv"
: > "$MANIFEST"
n_combos=0 n_skipped=0
chk() { [ -e "$1" ] && return 0; printf '    \033[31mmissing\033[0m %s\n' "$1"; return 1; }

# Append this combo's SHARDS rows to the manifest, or skip it if any asset is missing.
# Row = ACCENT \t SPEAKER \t REF_KIND \t REF_AUDIO \t REF_TEXT \t RUN_DIR \t SHARD_IDX \t SHARD_COUNT.
# REF_TEXT is a file path (romanised for hi/bn) -> no tabs/commas, safe in TSV and -v.
emit() {  # accent kind speaker
  local accent="$1" kind="$2" spk="$3" name="$accent/$kind/$spk"
  local run_dir; run_dir="$(run_dir_for "$accent")"
  if [ -z "$run_dir" ]; then printf '  \033[31mSKIP\033[0m %s (unknown accent)\n' "$name"; n_skipped=$((n_skipped+1)); return; fi

  local miss=0 ref_audio ref_text
  chk "$run_dir/config.yaml" || miss=1
  chk "$run_dir/vocab.txt"   || miss=1
  chk "vectors/$accent.pt"   || miss=1
  if [ "$kind" = l1 ]; then
    ref_audio="prompts/$accent/${accent}_${spk}.wav"; ref_text="prompts/$accent/${accent}_${spk}_ref.txt"
  else
    ref_audio="${NATIVE_PREFIX}_${spk}.wav"; ref_text="${NATIVE_PREFIX}_${spk}.txt"
  fi
  chk "$ref_audio" || miss=1
  chk "$ref_text"  || miss=1
  if [ "$miss" = 1 ]; then printf '  \033[31mSKIP\033[0m %s (assets above)\n' "$name"; n_skipped=$((n_skipped+1)); return; fi

  local s
  for ((s=0; s<SHARDS; s++)); do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$accent" "$spk" "$kind" "$ref_audio" "$ref_text" "$run_dir" "$s" "$SHARDS" >> "$MANIFEST"
  done
  printf '  \033[32mOK\033[0m   %s (%d shard-task(s))\n' "$name" "$SHARDS"
  n_combos=$((n_combos+1))
}

for accent in $ACCENTS; do for kind in $REF_KINDS; do for spk in $SPEAKERS; do
  emit "$accent" "$kind" "$spk"
done; done; done

N=$(wc -l < "$MANIFEST" | tr -d ' ')
echo "manifest: $MANIFEST  ($n_combos combos x $SHARDS shard(s) = $N tasks; $n_skipped combo(s) skipped)"
if [ "$N" -eq 0 ]; then
  echo "no runnable tasks (all assets missing); prepare them and re-run." >&2; rm -f "$MANIFEST"; exit 1
fi

QSUB=(qsub -t "1-$N" -tc "$MAX_CONCURRENT" -v "ALPHAS=$ALPHAS" scripts/eddie_infer_array.sh "$ACCENT_DIR/$MANIFEST")
if [ "${DRY_RUN:-0}" = 1 ]; then
  echo "--- manifest rows ---"; cat "$MANIFEST"
  echo "--- would submit ---"; printf '  %q ' "${QSUB[@]}"; printf '\n'
else
  "${QSUB[@]}"
  echo "submitted array of $N tasks (-tc $MAX_CONCURRENT). qstat to watch; qdel <jobid> to cancel all."
fi
[ "$n_skipped" -eq 0 ] || echo "note: $n_skipped combo(s) skipped for missing assets (see above)." >&2

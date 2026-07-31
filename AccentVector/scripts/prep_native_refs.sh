#!/bin/bash
# Prepare the neutral-English `native` reference clips (the decoupling control) from VCTK.
# Default: male p360_011, female p361_011 (both American / New Jersey per speaker-info.txt),
# mic1 -> prompts/GAE/gae_{m,f}.{wav,txt} (mono, 24 kHz, 16-bit PCM + one-line transcript).
#
# VCTK is ENGLISH, so these serve REF_KIND=native ONLY; the `l1` condition still needs
# native-LANGUAGE clips (Dutch/Hindi/Bengali). To use these in the sweep, point the submitter
# at them:  NATIVE_PREFIX=prompts/GAE/gae bash scripts/submit_infer_sweeps.sh
#
# Run on a login node with ffmpeg (module load ffmpeg / a conda env that has it):
#   bash scripts/prep_native_refs.sh
#   MALE=p294 FEMALE=p334 PREFIX=gae2 bash scripts/prep_native_refs.sh   # other speakers
set -euo pipefail

ACCENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VCTK_ROOT=${VCTK_ROOT:-"$ACCENT_DIR/../Datasets/vctk/VCTK-Corpus-0.92"}
OUT_DIR=${OUT_DIR:-"$ACCENT_DIR/prompts/GAE"}
PREFIX=${PREFIX:-gae}    # output basenames <PREFIX>_m / <PREFIX>_f (submitter: NATIVE_PREFIX=<out>/<PREFIX>)
MALE=${MALE:-p360}       # American male   (New Jersey)
FEMALE=${FEMALE:-p361}   # American female (New Jersey)
MIC=${MIC:-mic1}         # mic1 = omni (DPA), cleaner + always present; mic2 differs/absent for some speakers
SR=${SR:-24000}          # F5's target rate (F5 also resamples refs, so this is just tidiness)
UTT=${UTT:-011}          # 011 = the shared Rainbow-Passage sentence -> identical ref text across speakers

command -v ffmpeg >/dev/null || { echo "ffmpeg not found (module load ffmpeg / activate an env with it)" >&2; exit 1; }
mkdir -p "$OUT_DIR"

prep() {  # speaker_id  tag(m|f)
  local spk="$1" tag="$2"
  local flac="$VCTK_ROOT/wav48_silence_trimmed/$spk/${spk}_${UTT}_${MIC}.flac"
  local txt="$VCTK_ROOT/txt/$spk/${spk}_${UTT}.txt"
  [ -f "$flac" ] || { echo "missing audio: $flac  (VCTK 0.92 layout? adjust if not)" >&2; exit 1; }
  [ -f "$txt" ]  || { echo "missing transcript: $txt" >&2; exit 1; }
  ffmpeg -y -loglevel error -i "$flac" -ac 1 -ar "$SR" -sample_fmt s16 "$OUT_DIR/${PREFIX}_${tag}.wav"
  tr '\r\n' '  ' < "$txt" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//' > "$OUT_DIR/${PREFIX}_${tag}.txt"
  printf '  %s_%-2s <- %s_%s_%s   text: %s\n' "$PREFIX" "$tag" "$spk" "$UTT" "$MIC" "$(cat "$OUT_DIR/${PREFIX}_${tag}.txt")"
}

echo "VCTK -> neutral-English native refs (mic=$MIC, sr=$SR Hz) in $OUT_DIR:"
prep "$MALE" m
prep "$FEMALE" f
echo "done. Use with: NATIVE_PREFIX=${OUT_DIR#$ACCENT_DIR/}/$PREFIX bash scripts/submit_infer_sweeps.sh"

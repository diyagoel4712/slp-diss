#!/bin/bash
# Write a one-file provenance record for a fine-tune run: code versions + config,
# to stdout (-> the job log) AND a timestamped file under <out-dir>.
# Called by eddie_finetune_lora.sh; run config is read from the environment.
#
#   record_provenance.sh <accent_dir> <f5_root> <out_dir>
#
# Why not just a git SHA for the fork? F5-TTS is a gitignored clone with no .git
# (see F5-TTS/PROVENANCE.md), so it isn't version-controlled. The AccentVector SHA
# pins OUR code (incl. patch_f5_tts_fork.py + the accent config = the dissertation
# deltas); the fork's src/ is fingerprinted instead, so a changed clone or changed
# patches show up as a different hash.
set -euo pipefail

ACCENT_DIR="$1"; F5_ROOT="$2"; OUT_DIR="$3"

git_desc()  { git -C "$1" rev-parse HEAD 2>/dev/null || echo "n/a"; }
git_dirty() { test -n "$(git -C "$1" status --porcelain 2>/dev/null)" && echo "DIRTY" || echo "clean"; }
fork_fp() {
    python - "$1" <<'PY' 2>/dev/null || echo "n/a"
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
h = hashlib.sha256()
for p in sorted(root.rglob("*.py")):
    h.update(p.relative_to(root).as_posix().encode()); h.update(p.read_bytes())
print(h.hexdigest()[:16])
PY
}

mkdir -p "$OUT_DIR"
OUT_FILE="$OUT_DIR/ft.${JOB_ID:-local}.$(date +%Y%m%d_%H%M%S).txt"
{
    echo "job=${JOB_ID:-local}  host=$(hostname)  date=$(date +%Y-%m-%dT%H:%M:%S%z)"
    echo "accent=${ACCENT_NAME:-?}  logger=${LOGGER:-?}"
    echo "AccentVector: $(git_desc "$ACCENT_DIR") ($(git_dirty "$ACCENT_DIR"))"
    echo "F5-TTS fork:  src-fp=$(fork_fp "$F5_ROOT/src")  (gitignored clone; see F5-TTS/PROVENANCE.md)"
    echo "PRETRAIN=${PRETRAIN:-?}"
    echo "METADATA_CSV=${METADATA_CSV:-?}"
    echo "AUDIO_ROOT=${AUDIO_ROOT:-?}"
    echo "CKPT_ROOT=${CKPT_ROOT:-?}"
    echo "conda_env=${CONDA_DEFAULT_ENV:-?}  python=$(command -v python)"
    python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" 2>/dev/null || true
} | tee "$OUT_FILE"

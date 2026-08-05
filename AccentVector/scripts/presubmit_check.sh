#!/bin/bash
# Pre-submit sanity check for eddie_finetune_lora.sh. Run on a LOGIN NODE in the
# f5-tts env BEFORE qsub, so missing files/imports fail here (seconds) instead of
# after the GPU job queues and starts.
#   conda activate f5-tts && bash scripts/presubmit_check.sh
# Uses the SAME env-var defaults as eddie_finetune_lora.sh; override to match.
set -uo pipefail

ACCENT_DIR=$(cd "$(dirname "$0")/.." && pwd)
F5_ROOT=${F5_ROOT:-"$ACCENT_DIR/../F5-TTS"}
ACCENT_NAME=${ACCENT_NAME:-dutch}
METADATA_CSV=${METADATA_CSV:-/exports/eddie/scratch/s2247837/data/cgn_dutch_clips/metadata.dnsmos.csv}
AUDIO_ROOT=${AUDIO_ROOT:-/exports/eddie/scratch/s2247837/data/cgn_dutch_clips}
PRETRAIN=${PRETRAIN:-"$F5_ROOT/ckpts/F5TTS_v1_Base/model_1250000.pt"}
export F5_VOCAB=${F5_VOCAB:-"$F5_ROOT/examples/vocab.txt"}
CKPT_ROOT=${CKPT_ROOT:-/exports/eddie/scratch/s2247837/accentvector-exps}

fail=0
ok()  { printf '  \033[32mOK\033[0m   %s\n' "$1"; }
bad() { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=1; }
chk_file() { [ -f "$1" ] && ok "$2" || bad "$2 (missing: $1)"; }
chk_dir()  { [ -d "$1" ] && ok "$2" || bad "$2 (missing: $1)"; }

echo "== paths =="
chk_dir  "$F5_ROOT/src/f5_tts"                 "F5-TTS repo ($F5_ROOT)"
chk_file "$PRETRAIN"                           "base checkpoint"
chk_file "$F5_VOCAB"                           "base vocab (F5_VOCAB)"
chk_file "$ACCENT_DIR/scripts/F5TTS_v1_LoRA_accent.yaml" "LoRA config yaml"
chk_dir  "$AUDIO_ROOT/wavs"                    "audio clips dir"

echo "== dataset manifest =="
if [ -f "$METADATA_CSV" ]; then
    n=$(($(wc -l < "$METADATA_CSV") - 1))       # minus header
    [ "$n" -gt 0 ] && ok "metadata rows: $n" || bad "metadata has no data rows: $METADATA_CSV"
    # verify the first referenced clip actually exists on disk
    first=$(tail -n +2 "$METADATA_CSV" | head -1 | cut -d'|' -f1)
    chk_file "$AUDIO_ROOT/$first"               "first referenced clip resolves ($first)"
else
    bad "metadata missing: $METADATA_CSV  (run dnsmos_filter.py first)"
fi

echo "== writable output =="
if mkdir -p "$CKPT_ROOT" 2>/dev/null && [ -w "$CKPT_ROOT" ]; then ok "CKPT_ROOT writable: $CKPT_ROOT"; else bad "CKPT_ROOT not writable: $CKPT_ROOT"; fi
if mkdir -p "$ACCENT_DIR/logs" 2>/dev/null && [ -w "$ACCENT_DIR/logs" ]; then ok "logs/ writable (qsub -o)"; else bad "logs/ not writable: $ACCENT_DIR/logs"; fi

echo "== python env =="
if python - <<'PY'
import sys, os
try:
    import torch, torchaudio, f5_tts, accent_vector          # noqa: F401
    from accent_vector.data_preprocess import PRETRAINED_VOCAB_PATH
except Exception as e:
    print("  \033[31mFAIL\033[0m imports:", e); sys.exit(1)
print("  \033[32mOK\033[0m   imports: f5_tts, accent_vector, torch, torchaudio")
vp = str(PRETRAINED_VOCAB_PATH)
if os.path.exists(vp):
    print(f"  \033[32mOK\033[0m   prepare will read vocab: {vp}")
else:
    print(f"  \033[31mFAIL\033[0m prepare vocab path missing: {vp}"); sys.exit(1)
print(f"  INFO torch.cuda.is_available() = {torch.cuda.is_available()} "
      f"(False on a login node is expected; the GPU appears in the job)")
PY
then :; else fail=1; fi

echo
if [ "$fail" -eq 0 ]; then
    printf '\033[32mALL CHECKS PASSED\033[0m -- safe to: qsub scripts/eddie_finetune_lora.sh\n'
else
    printf '\033[31mSOME CHECKS FAILED\033[0m -- fix the above before qsub.\n'; exit 1
fi

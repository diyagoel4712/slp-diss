#!/bin/bash
# Per-accent L1 prompt lookup, shared by every submit_*.sh.
#
# The prompt basenames are NOT uniform -- only Dutch is <accent>_<spk>; the others
# keep the source corpus's speaker id (IndicVoices hi_/bn_, GlobalPhone ar_, FLEURS
# mandarin_). This table was previously duplicated in submit_eval_grid.sh and
# submit_indic_ckpt_grid.sh (with a "must match" comment between them) and hardcoded
# a third time in submit_dutch_ckpt_grid.sh, so a new speaker had to be added in
# three places or the manifests silently disagreed.
#
#   . "$(dirname "$0")/prompt_refs.sh"
#   base=$(l1base dutch f)      -> data/prompts/dutch/dutch_f
#   "$base.wav"  "${base}_ref.txt"
#
# Paths are relative to AccentVector/ (every submit script cds there first).

# l1base <accent> <speaker>  -> prompt basename, no extension. Empty if unknown.
l1base() {
  case "$1/$2" in
    dutch/m)    echo data/prompts/dutch/dutch_m;;
    dutch/f)    echo data/prompts/dutch/dutch_f;;
    hindi/m)    echo data/prompts/hindi/hi_M_04;;
    hindi/f)    echo data/prompts/hindi/hi_F_02;;
    bengali/m)  echo data/prompts/bengali/bn_M_01;;
    bengali/f)  echo data/prompts/bengali/bn_F_02;;
    # Arabic: held-out GlobalPhone speakers (extract_gp_prompts.py); GP romanised ref-text.
    arabic/m)   echo data/prompts/arabic/ar_M_AR010;;
    arabic/f)   echo data/prompts/arabic/ar_F_AR002;;
    # Mandarin: held-out FLEURS speakers (Hanzi ref-text; F5 pinyin-converts at infer).
    mandarin/m) echo data/prompts/mandarin/mandarin_M_824;;
    mandarin/f) echo data/prompts/mandarin/mandarin_F_369;;
  esac
}

# The neutral GAE control prompt, shared across accents: <prefix>_<spk>.{wav,txt}
NATIVE_PREFIX=${NATIVE_PREFIX:-data/prompts/GAE/gae}

# gdir <speaker> -> ground-truth gender dir under data/ground_truth_refs/<accent>/
gdir() { [ "$1" = f ] && echo female || echo male; }

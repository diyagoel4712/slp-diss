#!/usr/bin/env python3
"""Apply AccentVector's fixes to the F5-TTS fork (the LoRA-capable
the-bird-F/Expressive-Vectors fork, see F5-TTS/PROVENANCE.md). The fork lives
outside this repo (gitignored, reconstructed per README.md) so these fixes
can't travel with a normal git commit -- run this once, right after cloning
the fork, on any new machine (HPC included).

Idempotent: safe to re-run. Each patch is a string search-and-replace; if the
target string is already gone (patch previously applied) it's skipped, and if
neither the old nor new string is found the fork's source has diverged from
what these patches expect -- that patch is skipped with a warning so you can
apply it by hand instead of silently leaving broken source.

Usage
-----
    python scripts/patch_f5_tts_fork.py [--f5-root /path/to/F5-TTS]

Bugs fixed (all in F5-TTS/src, discovered running the toy LoRA pipeline on a
Mac -- see AccentVector conversation history for full crash traces):

1. F5TTS_v1_LoRA_accent.yaml: `lora_feature_dim: None` is YAML's string
   "None", not a null -- crashes range() in LoRALinear.__init__.
2. finetune_cli.py: `to_absolute_path(config.model.vocoder.local_path)` is
   called unconditionally; local_path is legitimately None whenever
   `is_local: False` (the default), and Path(None) crashes.
3. finetune_cli.py: `config.datasets.num_workers` was read from the YAML
   config but never forwarded into `trainer.train(...)`, so the value was
   always silently ignored in favor of Trainer.train's hardcoded default of 16.
4. trainer.py: `persistent_workers=True` was hardcoded in all 4 DataLoader
   constructions; PyTorch rejects that when num_workers=0.
5. dit.py: `self.lora_map` is only populated when `lora_feature_dim` is set
   (the multi-accent-per-run recipe). AccentVector's single-accent recipe
   (F5TTS_v1_LoRA_accent.yaml, lora_feature_dim=null) leaves it empty by
   design, but still tags every sample with a real lora_label -- so
   `self.lora_map[i]` KeyErrors on that empty dict. LoRALinear.forward
   already ignores the index in the single-branch case, so the fix just
   skips the lookup instead of computing an unused index into nothing.
"""

import argparse
import os
import sys

PATCHES = [
    dict(
        file="src/f5_tts/configs/F5TTS_v1_LoRA_accent.yaml",
        desc="lora_feature_dim: None (string) -> null (YAML null)",
        old="    lora_feature_dim: None  # single LoRA per run (one accent -> one vector)",
        new="    lora_feature_dim: null  # single LoRA per run (one accent -> one vector)",
    ),
    dict(
        file="src/f5_tts/train/finetune_cli.py",
        desc="guard to_absolute_path() against vocoder.local_path=None",
        old='        is_local_vocoder=config.model.vocoder.is_local,\n'
            '        local_vocoder_path=to_absolute_path(config.model.vocoder.local_path),',
        new='        is_local_vocoder=config.model.vocoder.is_local,\n'
            '        local_vocoder_path=to_absolute_path(config.model.vocoder.local_path)\n'
            '        if config.model.vocoder.local_path\n'
            '        else "",',
    ),
    dict(
        file="src/f5_tts/train/finetune_cli.py",
        desc="forward config.datasets.num_workers into trainer.train()",
        old='    trainer.train(\n'
            '        train_dataset,\n'
            '        valid_dataset,\n'
            '        resumable_with_seed=666,  # seed for shuffling dataset\n'
            '    )',
        new='    trainer.train(\n'
            '        train_dataset,\n'
            '        valid_dataset,\n'
            '        num_workers=config.datasets.get("num_workers", 16),\n'
            '        resumable_with_seed=666,  # seed for shuffling dataset\n'
            '    )',
    ),
    dict(
        file="src/f5_tts/model/trainer.py",
        desc="persistent_workers=True -> num_workers > 0 (all 4 DataLoaders)",
        old="persistent_workers=True,",
        new="persistent_workers=num_workers > 0,",
        replace_all=True,
    ),
    dict(
        file="src/f5_tts/model/backbones/dit.py",
        desc="skip empty self.lora_map lookup in single-LoRA-per-run mode",
        old="                x = block(x, t, mask=mask, rope=rope, "
            "lora_idx=lora_idx[self.lora_map[i]] if lora_idx is not None else None)",
        new="                # self.lora_map is only populated when lora_feature_dim is set (multi-branch\n"
            "                # LoRA); in the single-LoRA-per-run case (lora_feature_dim=None) it's {} and\n"
            "                # LoRALinear.forward ignores the index anyway, so skip the lookup rather than\n"
            "                # KeyError on data that still carries a (irrelevant) per-sample lora_idx.\n"
            "                block_lora_idx = lora_idx[self.lora_map[i]] if (lora_idx is not None and self.lora_map) else None\n"
            "                x = block(x, t, mask=mask, rope=rope, lora_idx=block_lora_idx)",
    ),
]


def apply_patch(f5_root, patch):
    path = os.path.join(f5_root, patch["file"])
    if not os.path.isfile(path):
        print(f"[SKIP] {patch['file']}: file not found under {f5_root}")
        return "missing"

    with open(path, encoding="utf-8") as f:
        content = f.read()

    if patch["new"] in content and patch["old"] not in content:
        print(f"[OK]   {patch['file']}: already patched ({patch['desc']})")
        return "already"

    count = content.count(patch["old"])
    if count == 0:
        print(f"[WARN] {patch['file']}: expected old text not found -- fork source has "
              f"diverged from what this patch expects ({patch['desc']}); apply by hand.")
        return "diverged"

    if not patch.get("replace_all") and count > 1:
        print(f"[WARN] {patch['file']}: old text matched {count} times, expected 1 "
              f"({patch['desc']}); apply by hand to avoid an ambiguous replace.")
        return "ambiguous"

    new_content = content.replace(patch["old"], patch["new"])
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_content)
    n = count if patch.get("replace_all") else 1
    print(f"[APPLIED] {patch['file']}: {patch['desc']} ({n} occurrence{'s' if n != 1 else ''})")
    return "applied"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--f5-root", default=os.environ.get("F5_ROOT") or
                     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "F5-TTS"))
    args = ap.parse_args()
    f5_root = os.path.abspath(args.f5_root)

    print(f"Patching F5-TTS fork at {f5_root}\n")
    results = [apply_patch(f5_root, p) for p in PATCHES]

    n_diverged = results.count("diverged") + results.count("ambiguous")
    n_missing = results.count("missing")
    print(f"\n{results.count('applied')} applied, {results.count('already')} already patched, "
          f"{n_diverged} need manual attention, {n_missing} file(s) not found.")
    if n_diverged or n_missing:
        sys.exit(1)


if __name__ == "__main__":
    main()

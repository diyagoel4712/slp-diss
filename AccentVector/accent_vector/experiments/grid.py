"""[A1] Build the synthesis grid: for each accent, an alpha sweep over the shared
held-out English transcripts, using that accent's **native-language (L1) reference**
clip -- the paper's cloning setup. The reference is held FIXED per accent across the
sweep, so within an accent only alpha varies:

    alpha=0   the pretrained model (no fine-tuning) cloning the accent from the
              reference alone -- exactly theta_pre.
    alpha=1   the fully fine-tuned model -- theta_pre + tau (the full accent vector).

So the sweep interpolates from "how much accent the base model copies from the
reference" up to "the full fine-tuning impact"; speaker identity should hold across it.

Reads a small JSON config, then invokes ``accent_vector.infer_accent`` per accent
(subprocess, so no GPU import until a sweep runs). Downstream analyses read
results/<accent>/alpha_<a>/utt####.wav.

Vector track (--lora): with --lora, "accents" map to LoRA vectors and the sweep
scales the LoRA branch natively (no merge); needs "lora_config" + "vocab" from the
training run, and an optional "lora_mapping" (accent name is passed as the label).
Without --lora, "accents" are full-weight vectors and the sweep merges per alpha.

Config (JSON):
    {
      "pretrained": ".../model_1250000.pt",
      "transcripts": "transcripts/eval_transcripts.txt",
      "alphas": "0,0.2,0.4,0.6,0.8,1.0",
      "out_root": "results",
      "accents": {"british": "vectors/british.pt", "spanish": "vectors/spanish.pt"},
      "references": {                               # native-language (L1) reference per accent
        "british": {"audio": "refs/england_L1.wav", "text": "..."},
        "spanish": {"audio": "refs/spanish_L1.wav", "text": "..."}
      },
      "lora_config": "exps/.../config.yaml",        # --lora only
      "vocab": "exps/.../vocab.txt",                # --lora only
      "lora_mapping": "exps/.../lora_mapping.json"  # --lora only, optional
    }

    python -m accent_vector.experiments.grid --config grid.json [--lora] [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _reference_for(accent, cfg):
    """(ref_audio, ref_text) -- this accent's native-language (L1) reference clip,
    held fixed across the accent's alpha sweep."""
    refs = cfg.get("references", {})
    if accent not in refs:
        raise SystemExit(
            f"references['{accent}'] (audio+text) missing from the config; each "
            f"accent needs its native-language (L1) reference clip"
        )
    r = refs[accent]
    return r["audio"], r["text"]


def _sweep_cmd(accent, vector, cfg, ref_audio, ref_text, out_dir, lora):
    """infer_accent invocation for one accent -- LoRA (native scaling) or merged."""
    common = [
        sys.executable, "-m", "accent_vector.infer_accent",
        "--pretrained", cfg["pretrained"],
        "--alphas", cfg.get("alphas", "0,0.2,0.4,0.6,0.8,1.0"),
        "--ref-audio", ref_audio,
        "--ref-text", ref_text,
        "--transcripts", cfg["transcripts"],
        "--out-dir", out_dir,
    ]
    if not lora:
        return common + ["--vector", vector]
    if "lora_config" not in cfg or "vocab" not in cfg:
        raise SystemExit("--lora needs 'lora_config' and 'vocab' (training run) in the config")
    cmd = common + ["--lora", "--lora-vector", vector,
                    "--config", cfg["lora_config"], "--vocab", cfg["vocab"]]
    if cfg.get("lora_mapping"):  # accent name is the label into lora_mapping.json
        cmd += ["--lora-mapping", cfg["lora_mapping"], "--lora-label", accent]
    return cmd


def build_grid(cfg, dry_run=False, lora=False):
    out_root = Path(cfg.get("out_root", "results"))
    track = "lora" if lora else "merged"
    for accent, vector in cfg["accents"].items():
        ref_audio, ref_text = _reference_for(accent, cfg)
        cmd = _sweep_cmd(accent, vector, cfg, ref_audio, ref_text,
                         str(out_root / accent), lora)
        print(f"[grid:{track}] {accent} (ref={ref_audio}): {' '.join(cmd)}")
        if not dry_run:
            subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Build the accent x alpha synthesis grid")
    parser.add_argument("--config", required=True, help="JSON grid config")
    parser.add_argument("--lora", action="store_true",
                        help="native-LoRA track: 'accents' are LoRA vectors, scaled in "
                             "place per alpha (needs lora_config + vocab in the config)")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    args = parser.parse_args()
    build_grid(json.loads(Path(args.config).read_text()),
               dry_run=args.dry_run, lora=args.lora)


if __name__ == "__main__":
    main()

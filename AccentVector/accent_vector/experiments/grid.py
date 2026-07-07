"""[A1/A3] Build the synthesis grid: for each accent, an alpha sweep over the
shared held-out English transcripts.

Two reference modes -- the difference between them is the reference-leakage
ablation (how much accent comes from the reference vs. the vector):

  fixed    one neutral reference for every accent and every alpha, so the accent
           vector is the only thing that varies (the default; clean attribution).
  matched  a per-accent L1 reference (replicating the paper's target-language
           pairing). Comparing matched vs fixed at alpha=0 isolates how much
           accent the reference alone supplies -- see control A3 in EXPERIMENTS.md.

Reads a small JSON config, then invokes ``accent_vector.infer_accent`` per accent
(subprocess, so no GPU import until a sweep runs). Downstream analyses read
results/<accent>/alpha_<a>/utt####.wav (matched runs go to <accent>__matchedref).

Config (JSON):
    {
      "pretrained": ".../model_1250000.pt",
      "ref_audio": "refs/neutral.wav",              # fixed-mode reference
      "ref_text": "Some call me nature ...",
      "transcripts": "transcripts/eval_transcripts.txt",
      "alphas": "0,0.2,0.4,0.6,0.8,1.0",
      "out_root": "results",
      "accents": {"british": "vectors/british.pt", "spanish": "vectors/spanish.pt"},
      "references": {                               # matched-mode references (per accent)
        "british": {"audio": "refs/england.wav", "text": "..."},
        "spanish": {"audio": "refs/spanish.wav", "text": "..."}
      }
    }

    python -m accent_vector.experiments.grid --config grid.json [--reference-mode matched] [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _reference_for(accent, cfg, mode):
    """Return (ref_audio, ref_text, out_subdir) for this accent under the mode."""
    if mode == "matched":
        refs = cfg.get("references", {})
        if accent not in refs:
            raise SystemExit(
                f"--reference-mode matched needs references['{accent}'] "
                f"(audio+text) in the config"
            )
        r = refs[accent]
        return r["audio"], r["text"], f"{accent}__matchedref"
    # fixed: one neutral reference for all accents
    if "ref_audio" not in cfg or "ref_text" not in cfg:
        raise SystemExit("fixed reference mode needs ref_audio + ref_text in the config")
    return cfg["ref_audio"], cfg["ref_text"], accent


def build_grid(cfg, mode="fixed", dry_run=False):
    out_root = Path(cfg.get("out_root", "results"))
    for accent, vector in cfg["accents"].items():
        ref_audio, ref_text, out_subdir = _reference_for(accent, cfg, mode)
        cmd = [
            sys.executable, "-m", "accent_vector.infer_accent",
            "--pretrained", cfg["pretrained"],
            "--vector", vector,
            "--alphas", cfg.get("alphas", "0,0.2,0.4,0.6,0.8,1.0"),
            "--ref-audio", ref_audio,
            "--ref-text", ref_text,
            "--transcripts", cfg["transcripts"],
            "--out-dir", str(out_root / out_subdir),
        ]
        print(f"[grid:{mode}] {accent} (ref={ref_audio}): {' '.join(cmd)}")
        if not dry_run:
            subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Build the accent x alpha synthesis grid")
    parser.add_argument("--config", required=True, help="JSON grid config")
    parser.add_argument("--reference-mode", choices=["fixed", "matched"], default="fixed",
                        help="fixed: one neutral reference for all accents (isolates the "
                             "vector); matched: per-accent L1 reference (paper-style; the "
                             "matched-vs-fixed gap measures reference leakage)")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    args = parser.parse_args()
    build_grid(json.loads(Path(args.config).read_text()),
               mode=args.reference_mode, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

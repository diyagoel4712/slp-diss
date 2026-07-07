"""[A1] Build the synthesis grid: for each accent, an alpha sweep over the shared
held-out English transcripts with a fixed neutral reference.

Reads a small JSON config listing accents and their vectors, then invokes
``accent_vector.infer_accent`` per accent (as a subprocess, so no GPU import
happens until a sweep actually runs). Every downstream analysis reads the tree
this produces: results/<accent>/alpha_<a>/utt####.wav.

Config (JSON):
    {
      "pretrained": ".../model_1250000.pt",
      "ref_audio": "refs/neutral.wav",
      "ref_text": "Some call me nature ...",
      "transcripts": "transcripts/eval_transcripts.txt",
      "alphas": "0,0.2,0.4,0.6,0.8,1.0",
      "out_root": "results",
      "accents": {"british": "vectors/british.pt", "spanish": "vectors/spanish.pt"}
    }

    python -m accent_vector.experiments.grid --config grid.json [--dry-run]
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_grid(cfg, dry_run=False):
    out_root = Path(cfg.get("out_root", "results"))
    for accent, vector in cfg["accents"].items():
        cmd = [
            sys.executable, "-m", "accent_vector.infer_accent",
            "--pretrained", cfg["pretrained"],
            "--vector", vector,
            "--alphas", cfg.get("alphas", "0,0.2,0.4,0.6,0.8,1.0"),
            "--ref-audio", cfg["ref_audio"],
            "--ref-text", cfg["ref_text"],
            "--transcripts", cfg["transcripts"],
            "--out-dir", str(out_root / accent),
        ]
        print(f"[grid] {accent}: {' '.join(cmd)}")
        if not dry_run:
            subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Build the accent x alpha synthesis grid")
    parser.add_argument("--config", required=True, help="JSON grid config")
    parser.add_argument("--dry-run", action="store_true", help="print commands only")
    args = parser.parse_args()
    build_grid(json.loads(Path(args.config).read_text()), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

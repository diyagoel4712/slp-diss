"""Completeness gate for a synthesis tree: flag any alpha_<a>/ dir whose utt####.wav
set doesn't match the expected transcript count -- catches a failed/partial task (e.g. a
crashed shard) BEFORE it silently biases rq1/rq2/rq3 (which just score whatever wavs exist).

Expected count per alpha dir:
  --expected N         use N for every dir (uniform transcript set), OR
  (default) infer      per-speaker: read <transcripts-dir>/*_<spk>_eval.txt line count, where
                       <spk> is the path component (m/f) of the alpha dir. Handles the Dutch
                       grid where transcripts differ per speaker (f=7, m=5).

Checks both the COUNT and the exact index set 0..expected-1, so a shard that dropped utt0003
is reported as missing [3] even if some other file inflates the count.

    python scripts/check_sweep_complete.py --root results/dutch
    python scripts/check_sweep_complete.py --root results/british --expected 20
Exit code is nonzero if any alpha dir is incomplete, so it can gate an eval script:
    python scripts/check_sweep_complete.py --root results/dutch && bash run_eval.sh
"""

import argparse
import re
import sys
from pathlib import Path


def transcript_counts(transcripts_dir):
    """{speaker_tag: n_nonblank_lines} from <dir>/*_<tag>_eval.txt (tag = token before _eval)."""
    out = {}
    d = Path(transcripts_dir)
    for f in sorted(d.glob("*_eval.txt")) if d.is_dir() else []:
        m = re.match(r".*_([^_]+)_eval$", f.stem)
        if not m:
            continue
        n = sum(1 for ln in f.read_text(encoding="utf-8").splitlines() if ln.strip())
        out[m.group(1)] = n
    return out


def utt_indices(alpha_dir):
    idx = []
    for w in Path(alpha_dir).glob("*.wav"):
        m = re.search(r"utt(\d+)", w.stem)
        if m:
            idx.append(int(m.group(1)))
    return sorted(idx)


def expected_for(alpha_dir, counts, override):
    if override is not None:
        return override, "override"
    # find a path component that names a known speaker tag
    for part in Path(alpha_dir).parts:
        if part in counts:
            return counts[part], f"{part}_eval.txt"
    return None, "unknown"


def main():
    ap = argparse.ArgumentParser(description="Flag incomplete alpha_<a>/ synthesis dirs")
    ap.add_argument("--root", required=True, help="results tree to scan (recurses for alpha_<a>/ dirs)")
    ap.add_argument("--transcripts-dir", default="transcripts/dutch",
                    help="dir of <accent>_<spk>_eval.txt for per-speaker expected counts")
    ap.add_argument("--expected", type=int, help="expected utt count for EVERY dir (skips inference)")
    a = ap.parse_args()

    counts = {} if a.expected is not None else transcript_counts(a.transcripts_dir)
    alpha_dirs = sorted(p for p in Path(a.root).rglob("alpha_*") if p.is_dir())
    if not alpha_dirs:
        print(f"[check] no alpha_<a>/ dirs under {a.root}", file=sys.stderr)
        return 2

    n_ok = n_bad = n_unknown = 0
    for d in alpha_dirs:
        exp, src = expected_for(d, counts, a.expected)
        idx = utt_indices(d)
        rel = d.relative_to(a.root)
        if exp is None:
            print(f"  ?? {rel}  ({len(idx)} wavs; no expected count -- pass --expected or fix --transcripts-dir)")
            n_unknown += 1
            continue
        missing = sorted(set(range(exp)) - set(idx))
        extra = sorted(set(idx) - set(range(exp)))
        if not missing and not extra:
            n_ok += 1
        else:
            n_bad += 1
            note = []
            if missing:
                note.append(f"missing {missing}")
            if extra:
                note.append(f"unexpected {extra}")
            print(f"  XX {rel}  expected {exp} ({src}), found {len(idx)} -> {', '.join(note)}")

    print(f"[check] {len(alpha_dirs)} alpha dirs: {n_ok} complete, {n_bad} incomplete, {n_unknown} unknown")
    return 1 if (n_bad or n_unknown) else 0


if __name__ == "__main__":
    sys.exit(main())

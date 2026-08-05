"""[eval add-on] Per-alpha UTMOS (naturalness MOS) over one sweep dir -> utmos.csv.

UTMOS lives in its own env (utmosv2 pins conflict with the .conda / genaid metric
envs), so it is scored as a standalone pass rather than folded into rq1/rq3. It is
reference-free (no ground-truth clips needed), so it runs for every accent.

Run with the utmos env's python:
    <utmos-env>/bin/python Evaluation/score_utmos.py \
        --sweep-dir results/dutch/native/f/step_73400 --out-csv <...>/utmos.csv
"""
import argparse
import csv
from pathlib import Path


def alpha_dirs(sweep_dir):
    """(alpha, dir) for each alpha_<a>/ under sweep_dir, sorted by alpha."""
    out = []
    for d in Path(sweep_dir).glob("alpha_*"):
        if not d.is_dir():
            continue
        try:
            out.append((float(d.name.split("_", 1)[1]), d))
        except (ValueError, IndexError):
            continue
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description="per-alpha UTMOS over a sweep dir")
    p.add_argument("--sweep-dir", required=True)
    p.add_argument("--out-csv", required=True)
    a = p.parse_args()

    import numpy as np
    import utmosv2

    model = utmosv2.create_model(pretrained=True)
    rows = []
    for alpha, d in alpha_dirs(a.sweep_dir):
        if not any(d.glob("*.wav")):
            continue
        preds = model.predict(input_dir=str(d))            # [{'file_path','predicted_mos'}]
        mos = [x["predicted_mos"] for x in preds]
        rows.append({"alpha": alpha,
                     "utmos": float(np.mean(mos)) if mos else float("nan"),
                     "n": len(mos)})
        print(f"[utmos] alpha={alpha}: {rows[-1]}")

    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["alpha", "utmos", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"[utmos] wrote {a.out_csv}")


if __name__ == "__main__":
    main()

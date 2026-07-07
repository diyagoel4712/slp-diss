"""[E3.4] RQ3 -- layer localisation of the accent vector.

Where in the network does the accent shift live? Partition the vector's tensors
into module groups (attention / feed-forward / conv / text-embed, and
early/late by block index) and report each group's share of the total RMS
magnitude. Groups that carry a large share are candidates to scale on their own
(via ``extract_vector compose --include ...``) to test their segmental vs
suprasegmental effect -- the ablation that localises where prosody lives.

    python -m accent_vector.experiments.rq3_layers \
        --vector vectors/british.pt --out-csv results/british/rq3_layers.csv
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np

# substring -> module group. Tuned for F5-TTS's DiT keys (attn / ff / conv /
# text embed); adjust if the backbone's naming differs.
MODULE_PATTERNS = {
    "attention": ["attn", "to_q", "to_k", "to_v", "to_out"],
    "feed_forward": ["ff.", "ff_", "mlp", "linear"],
    "conv": ["conv"],
    "text_embed": ["text_embed", "text_blocks", "embed"],
    "norm": ["norm"],
}


def _group(key):
    for name, subs in MODULE_PATTERNS.items():
        if any(s in key for s in subs):
            return name
    return "other"


def _block_index(key):
    m = re.search(r"blocks?[.@](\d+)", key)
    return int(m.group(1)) if m else None


def run(vector_path, out_csv):
    from accent_vector.extract_vector import load_flat_checkpoint

    flat = load_flat_checkpoint(vector_path)
    group_energy, depth_energy, total = {}, {}, 0.0
    max_block = 0
    per_key = []
    for key, t in flat.items():
        if not (hasattr(t, "is_floating_point") and t.is_floating_point()):
            continue
        rms = float(t.detach().cpu().float().pow(2).mean().sqrt())
        g = _group(key)
        group_energy[g] = group_energy.get(g, 0.0) + rms
        b = _block_index(key)
        if b is not None:
            max_block = max(max_block, b)
        per_key.append((key, g, b, rms))
        total += rms

    # early vs late by block index (split at the midpoint)
    mid = max_block / 2 if max_block else 0
    for _, _, b, rms in per_key:
        if b is not None:
            depth_energy["early" if b <= mid else "late"] = \
                depth_energy.get("early" if b <= mid else "late", 0.0) + rms

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["partition", "group", "rms_sum", "share"])
        for g, e in sorted(group_energy.items(), key=lambda kv: -kv[1]):
            w.writerow(["module", g, f"{e:.6f}", f"{e/total:.4f}" if total else "nan"])
        for g, e in sorted(depth_energy.items()):
            denom = sum(depth_energy.values())
            w.writerow(["depth", g, f"{e:.6f}", f"{e/denom:.4f}" if denom else "nan"])

    print("[rq3-layers] module shares:",
          {g: round(e / total, 3) for g, e in group_energy.items()} if total else {})
    print("[rq3-layers] depth shares:",
          {g: round(e / sum(depth_energy.values()), 3) for g, e in depth_energy.items()}
          if depth_energy else {})
    print(f"[rq3-layers] wrote {out_csv}")


def main():
    p = argparse.ArgumentParser(description="RQ3 layer localisation")
    p.add_argument("--vector", required=True)
    p.add_argument("--out-csv", required=True)
    a = p.parse_args()
    run(a.vector, a.out_csv)


if __name__ == "__main__":
    main()

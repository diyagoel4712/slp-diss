"""[E2.1-2.3] RQ2 -- geometry of the accent-vector space.

Builds two accent-similarity matrices and compares them:

    weight space  : cosine between the accent VECTORS themselves (cheap, no audio)
    output space  : cosine between GenAID accent EMBEDDINGS of synthesized clips

and runs a Mantel (RSA) test between them (H2: positive but imperfect). A 2-D
classical-MDS embedding of the weight-space matrix gives the accent map.

    python -m accent_vector.experiments.rq2_geometry \
        --vector british=vectors/british.pt --vector spanish=vectors/spanish.pt \
        --synth british=results/british/alpha_1.0 --synth spanish=results/spanish/alpha_1.0 \
        --out-dir results/geometry
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from accent_vector.experiments import common


def _parse_pairs(items):
    out = {}
    for it in items:
        name, path = it.split("=", 1)
        out[name] = path
    return out


def weight_space_matrix(vectors, include=None, exclude=None):
    loaded = {n: common.load_vector_flat(p, include=include, exclude=exclude)
              for n, p in vectors.items()}
    return common.cosine_matrix(loaded)


def output_space_matrix(synth_dirs):
    """Mean GenAID accent embedding per accent -> cosine matrix."""
    ef = common.load_eval()
    names = list(synth_dirs)
    centroids = {}
    for name in names:
        wavs = common.wavs_in(synth_dirs[name])
        preds = ef.predict_accent_genaid(wavs, with_embeddings=True)
        embs = np.asarray([p["embedding"] for p in preds], dtype=float)
        centroids[name] = embs.mean(axis=0)
    return common.cosine_matrix(centroids)


def _write_matrix(path, names, M):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, n in enumerate(names):
            w.writerow([n] + [f"{M[i, j]:.4f}" for j in range(len(names))])


def run(vectors, synth_dirs, out_dir, include, exclude):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names_w, W = weight_space_matrix(vectors, include=include, exclude=exclude)
    _write_matrix(out_dir / "weight_space_cosine.csv", names_w, W)
    coords = common.classical_mds(W, 2)
    with open(out_dir / "weight_space_mds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["accent", "mds1", "mds2"])
        for n, (x, y) in zip(names_w, coords):
            w.writerow([n, f"{x:.4f}", f"{y:.4f}"])
    print(f"[rq2] weight-space matrix + MDS over {names_w}")

    if synth_dirs:
        common_names = [n for n in names_w if n in synth_dirs]
        names_o, O = output_space_matrix({n: synth_dirs[n] for n in common_names})
        _write_matrix(out_dir / "output_space_cosine.csv", names_o, O)
        # align weight matrix to the same accent order before RSA
        idx = [names_w.index(n) for n in names_o]
        Wr = W[np.ix_(idx, idx)]
        if len(names_o) >= 3:
            r, p = common.mantel_test(Wr, O)
            (out_dir / "rsa_mantel.txt").write_text(
                f"accents: {names_o}\nMantel r={r:.4f} p={p:.4f}\n")
            print(f"[rq2] RSA Mantel r={r:.4f} p={p:.4f} over {names_o}")
        else:
            print("[rq2] need >=3 accents for a Mantel test; skipping RSA")


def main():
    p = argparse.ArgumentParser(description="RQ2 accent-vector geometry")
    p.add_argument("--vector", action="append", default=[], required=True,
                   help="name=path (repeatable)")
    p.add_argument("--synth", action="append", default=[],
                   help="name=alpha_1.0_dir (repeatable; enables output-space RSA)")
    p.add_argument("--include", action="append", default=[],
                   help="restrict vectors to layers containing this substring")
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    run(_parse_pairs(a.vector), _parse_pairs(a.synth), a.out_dir,
        a.include or None, a.exclude or None)


if __name__ == "__main__":
    main()

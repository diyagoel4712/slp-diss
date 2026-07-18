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

from accent_vector.experiments import shared


def load_vector_flat(vector_path, include=None, exclude=None):
    """Flatten a saved accent vector into one 1-D numpy array over its float
    tensors, in sorted-key order (stable across accents so vectors are
    comparable). Optional include/exclude substrings restrict to a layer subset.

    Accepts either track's vector: a full-weight diff from ``extract_vector`` or a
    LoRA snapshot (``lora_<step>.pt`` / a saved ``lora_state_dict``) -- both
    flatten through ``load_flat_checkpoint``, and a LoRA snapshot's keys keep the
    ``blocks.N.attn...`` paths so ``rq3_layers`` groups them the same way.

    NB: full fine-tune vectors are ~300M-D; prefer LoRA vectors here, or pass a
    layer filter, to keep this in memory.
    """
    from accent_vector.extract_vector import load_flat_checkpoint, _key_selected

    flat = load_flat_checkpoint(vector_path)
    parts = []
    for key in sorted(flat):
        t = flat[key]
        if hasattr(t, "is_floating_point") and t.is_floating_point():
            if _key_selected(key, include, exclude):
                parts.append(t.detach().cpu().float().numpy().ravel())
    if not parts:
        raise ValueError(f"no float tensors selected in {vector_path}")
    return np.concatenate(parts)


def mantel_test(A, B, n_perm=9999, seed=0):
    """Correlate two symmetric matrices over their upper triangle, with a row/col
    permutation test. Returns (pearson_r, p_value). Used for RSA between the
    weight-space and output-space accent-similarity matrices (RQ2)."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    iu = np.triu_indices_from(A, k=1)
    a = A[iu]
    r_obs = float(np.corrcoef(a, B[iu])[0, 1])
    rng = np.random.default_rng(seed)
    n = A.shape[0]
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if abs(np.corrcoef(a, B[np.ix_(p, p)][iu])[0, 1]) >= abs(r_obs):
            hits += 1
    return r_obs, (hits + 1) / (n_perm + 1)



def _parse_pairs(items):
    out = {}
    for it in items:
        name, path = it.split("=", 1)
        out[name] = path
    return out


def weight_space_matrix(vectors, include=None, exclude=None):
    loaded = {n: load_vector_flat(p, include=include, exclude=exclude)
              for n, p in vectors.items()}
    return shared.cosine_matrix(loaded)


def output_space_matrix(synth_dirs):
    """Mean GenAID accent embedding per accent -> cosine matrix."""
    ef = shared.load_eval()
    names = list(synth_dirs)
    centroids = {}
    for name in names:
        wavs = shared.wavs_in(synth_dirs[name])
        preds = ef.predict_accent_genaid(wavs, with_embeddings=True)
        embs = np.asarray([p["embedding"] for p in preds], dtype=float)
        centroids[name] = embs.mean(axis=0)
    return shared.cosine_matrix(centroids)


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
    coords = shared.classical_mds(W, 2)
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
            r, p = mantel_test(Wr, O)
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

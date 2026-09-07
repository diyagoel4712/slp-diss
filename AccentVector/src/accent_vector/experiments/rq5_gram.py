"""RQ5 go/no-go -- the accent-vector Gram matrix and what it says.

Everything in the weight-space arm of RQ5 (H5a) is a function of the K x K matrix
of inner products between accent vectors:

    G[i][j] = <tau_i, tau_j>

Given G you get, with no further model access: the cosine matrix, the eigenspectrum
(how many directions the accents actually occupy), the mu/residual split, and every
leave-one-out least-squares reconstruction. So this script answers "is there any
structure here at all" before a single wav is synthesized.

Two things make it cheap enough to run on a laptop.

*The vector is a low-rank product.* The fork's LoRA branch computes
``decoders(encoders(x)) * lora_alpha``, so the induced weight change is
``dW = lambda * B @ A`` (``extract_vector`` calls the same quantity the accent
vector). Crucially the geometry must be computed on dW and NOT on the raw
encoders/decoders tensors: for any invertible G the pair ``(GA, BG^-1)`` gives an
identical dW and an identical model, so a cosine over A/B measures the arbitrary
gauge that training happened to land in as much as it measures accent. Accents
trained in separate runs land in unrelated gauges.

*You never build dW.* Materialising would turn ~10M LoRA parameters into ~300M
(~1.2 GB per accent). Unnecessary -- the Frobenius inner product of two rank-r
deltas factors through two r x r matrices:

    <B_i A_i, B_j A_j> = tr( (B_i^T B_j) (A_j A_i^T) )

with r = 16, so each pair costs O(r^2 (in + out)) instead of O(in * out).

lambda (``lora_alpha``) is 1 for a vector at full strength -- alpha scaling happens
at inference via ``set_lora_alpha``, which scales the branch OUTPUT and is correct.

    WARNING, verified by test: never scale a raw LoRA snapshot elementwise (i.e.
    never scale one with ``set_lora_alpha``). Multiplying the state
    dict by ``s`` hits encoders and decoders both, so ``dW = B A`` picks up ``s^2``
    while ``decoders.bias`` picks up only ``s``. That is not a rescaled accent
    vector at all -- the matrix and bias parts move apart, so the result is not a
    scalar multiple of tau and no single coefficient can express it.

Usage
-----
    python -m accent_vector.experiments.rq5_gram \
        --vector british=vectors/british_lora.pt \
        --vector dutch=vectors/dutch_lora.pt \
        --vector hindi=vectors/hindi_lora.pt \
        --vector bengali=vectors/bengali_lora.pt \
        --family british=germanic --family dutch=germanic \
        --family hindi=indic --family bengali=indic \
        --replicate hindi=snapshots/hindi/lora_45000.pt \
        --out-dir results/geometry/gram

    # the tokenizer control: Mandarin's vector has mass in pinyin embedding rows
    # no Latin/romanised accent touches, which deflates its raw cosine for a
    # reason that is tokenisation and not phonology.
    ... --exclude text_embed --exclude input_embed --out-dir results/geometry/gram_notext

This module is CPU-only and deliberately does not import f5_tts.
"""

import argparse
import csv
import itertools
import re
from pathlib import Path

import numpy as np
import torch

from accent_vector.extract_vector import _key_selected

# encoders/decoders leaf, with an optional branch index when lora_feature_dim is set
_LORA_KEY = re.compile(
    r"^(?P<mod>.+?)\.(?P<part>encoders|decoders)\.(?:(?P<idx>\d+)\.)?(?P<leaf>weight|bias)$"
)


def load_lora_vector(path):
    """The trainable LoRA tensors from a snapshot/vector (unwraps ``lora_state_dict``).

    Light-weight twin of ``lora_model.load_lora_state`` that avoids importing F5.
    """
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "lora_state_dict" in obj:
        obj = obj["lora_state_dict"]
    if not isinstance(obj, dict):
        raise ValueError(f"expected a state dict in {path}, got {type(obj)}")
    return obj


def lora_factors(state, rank, include=None, exclude=None):
    """-> {module_key: (B, A, bias_or_None)} such that the induced delta is ``B @ A``.

    Normalises the fork's three LoRA flavours onto one shape, B: (m, r), A: (r, n):

      LoRALinear     encoders (r, in),        decoders (out, r)      -> B=D,  A=E
      LoRAConv1d     encoders (r, in, k),     decoders (out, r, 1)   -> B=D[:,:,0],
                                                                       A=E.reshape(r, -1)
                     (the decoder kernel is 1, so the composite conv kernel is just a
                     matmul over the rank axis and flattening (in, k) is exact)
      LoRAEmbedding  encoders (n_emb, r),     decoders (emb_dim, r)  -> B=E,  A=D.T

    ``decoders.bias`` (Linear/Conv only) is a genuine part of the vector and is
    carried alongside, contributing a plain dot product to the inner product.
    """
    parts = {}
    for key, tensor in state.items():
        m = _LORA_KEY.match(key)
        if m is None:
            continue
        if not _key_selected(key, include, exclude):
            continue
        mod = m["mod"] if m["idx"] is None else f"{m['mod']}#{m['idx']}"
        # float64 throughout: R^2 for a well-reconstructed fold is 1 minus a ratio of
        # nearly equal quantities, and in fp32 that cancellation costs ~4 decimal
        # places -- enough to make an exact combination look like R^2 = 0.9995.
        parts.setdefault(mod, {})[f"{m['part']}.{m['leaf']}"] = tensor.detach().double()

    out = {}
    for mod, p in parts.items():
        E, D = p.get("encoders.weight"), p.get("decoders.weight")
        if E is None or D is None:
            continue  # a lone bias or an unpaired half carries no delta on its own
        if D.ndim == 3:                                  # Conv1d
            B, A = D[:, :, 0], E.reshape(E.shape[0], -1)
        elif E.shape[0] == rank:                         # Linear
            B, A = D, E
        elif E.shape[1] == rank:                         # Embedding
            B, A = E, D.t()
        else:
            raise ValueError(
                f"{mod}: cannot locate the rank-{rank} axis in encoders {tuple(E.shape)} "
                f"/ decoders {tuple(D.shape)} -- pass the right --rank"
            )
        out[mod] = (B.contiguous(), A.contiguous(), p.get("decoders.bias"))
    if not out:
        raise ValueError("no LoRA modules survived the include/exclude filter")
    return out


def inner(fa, fb):
    """<tau_a, tau_b> over the shared modules, via tr((B_a^T B_b)(A_b A_a^T)).

    Never materialises B @ A. The bias adds a plain dot product.
    """
    total = 0.0
    for mod in fa.keys() & fb.keys():
        Ba, Aa, ba = fa[mod]
        Bb, Ab, bb = fb[mod]
        if Ba.shape[0] != Bb.shape[0] or Aa.shape[1] != Ab.shape[1]:
            continue  # different geometry (e.g. a vocab-extended run); skip, don't guess
        total += torch.einsum("ij,ji->", Ba.t() @ Bb, Ab @ Aa.t()).item()
        if ba is not None and bb is not None:
            total += torch.dot(ba, bb).item()
    return total


def gram(vectors):
    """Symmetric K x K matrix of inner products over {name: factors}."""
    names = list(vectors)
    K = len(names)
    G = np.zeros((K, K))
    for i, j in itertools.combinations_with_replacement(range(K), 2):
        G[i, j] = G[j, i] = inner(vectors[names[i]], vectors[names[j]])
    return names, G


def cosine_from_gram(G):
    d = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    return G / np.outer(d, d)


def loo_reconstruction(G, k):
    """Leave-one-out fit of tau_k from the other columns, entirely from G.

    Normal equations (T^T T) w = T^T tau_k -- both sides are already in G, so the
    whole oracle arm is arithmetic on a K x K table. Returns
    (coeffs, r2_full, r2_mean_only) where the mean-only model is the one-parameter
    fit tau_k ~ c * mu. The INCREMENT between them is the quantity H5a lives on: a
    random-direction null is worthless here (five random directions against a ~1e7
    dimensional target give R^2 ~ 1e-6, so it is cleared unconditionally), whereas
    the increment isolates accent-specific structure from the shared component.
    """
    idx = [i for i in range(G.shape[0]) if i != k]
    Gsub = G[np.ix_(idx, idx)]        # T^T T
    g = G[idx, k]                     # T^T tau_k
    ss_tot = G[k, k]

    w = np.linalg.pinv(Gsub) @ g
    r2_full = 1.0 - (ss_tot - g @ w) / ss_tot

    # mu = mean of the remaining vectors; <tau_k, mu> and <mu, mu> come from G too
    tk_mu = g.mean()
    mu_mu = Gsub.mean()
    r2_mean = (tk_mu ** 2) / (mu_mu * ss_tot) if mu_mu > 0 else 0.0

    return w, float(r2_full), float(r2_mean)


def residual_gram(G):
    """Double-centred Gram = inner products of r_i = tau_i - mu. Since sum_i r_i = 0,
    the off-diagonal cosines here average -1/(K-1) BY CONSTRUCTION -- comparing them
    against zero would 'discover' that accents are dissimilar when all that happened
    is a mean was subtracted. ``residual_cosine_baseline`` is that reference value.
    """
    K = G.shape[0]
    J = np.eye(K) - np.ones((K, K)) / K
    return J @ G @ J


def residual_cosine_baseline(K):
    return -1.0 / (K - 1)


def _pairs(items):
    out = {}
    for it in items:
        name, value = it.split("=", 1)
        out[name] = value
    return out


def _write_matrix(path, names, M, fmt="{:.6g}"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + names)
        for i, n in enumerate(names):
            w.writerow([n] + [fmt.format(M[i, j]) for j in range(len(names))])


def run(vector_paths, replicate_paths, families, rank, include, exclude, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gram] loading {len(vector_paths)} accent vectors (rank {rank})")
    vectors = {}
    for name, path in vector_paths.items():
        f = lora_factors(load_lora_vector(path), rank, include, exclude)
        vectors[name] = f
        print(f"        {name:<10} {len(f)} LoRA modules  <- {path}")

    names, G = gram(vectors)
    K = len(names)
    C = cosine_from_gram(G)
    norms = np.sqrt(np.diag(G))

    _write_matrix(out_dir / "gram.csv", names, G)
    _write_matrix(out_dir / "cosine.csv", names, C, fmt="{:.4f}")

    # ---- 1. magnitudes -----------------------------------------------------
    print("\n[gram] vector norms ||tau||")
    for n, v in zip(names, norms):
        print(f"        {n:<10} {v:12.4f}")
    with open(out_dir / "norms.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["accent", "norm"])
        w.writerows([[n, f"{v:.6g}"] for n, v in zip(names, norms)])

    # ---- 2. eigenspectrum: how many directions do the accents occupy? ------
    # The verdict runs on the COSINE matrix, not the raw Gram. The raw spectrum is
    # dominated by whichever accent simply has the largest ||tau||: a set of
    # perfectly orthogonal vectors with unequal norms still puts most of the raw
    # energy in one eigenvalue, which would read as "structured" when nothing is
    # aligned at all. Normalising first asks the question actually wanted -- how
    # aligned are the directions -- and makes the scale interpretable, since
    # trace(C) = K so an isotropic set gives every eigenvalue a share of 1/K.
    evals = np.sort(np.linalg.eigvalsh(C))[::-1]
    share = evals / evals.sum()
    Cc = residual_gram(C)                       # centred UNIT vectors
    evals_c = np.sort(np.linalg.eigvalsh(Cc))[::-1]
    share_c = evals_c / max(evals_c.sum(), 1e-30)
    evals_raw = np.sort(np.linalg.eigvalsh(G))[::-1]
    share_raw = evals_raw / evals_raw.sum()

    print("\n[gram] eigenspectrum, share of total")
    print("        i   cosine    after removing mu    raw Gram (norm-weighted)")
    for i in range(K):
        print(f"        {i+1}   {share[i]:7.4f}   {share_c[i]:16.4f}    {share_raw[i]:16.4f}")
    with open(out_dir / "eigenspectrum.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "eig_cosine", "share_cosine", "eig_centred", "share_centred",
                    "eig_raw_gram", "share_raw_gram"])
        for i in range(K):
            w.writerow([i + 1, f"{evals[i]:.6g}", f"{share[i]:.6f}",
                        f"{evals_c[i]:.6g}", f"{share_c[i]:.6f}",
                        f"{evals_raw[i]:.6g}", f"{share_raw[i]:.6f}"])

    # participation ratio: an effective count of occupied directions
    part_ratio = (evals.sum() ** 2) / (evals ** 2).sum()
    part_ratio_c = (evals_c.sum() ** 2) / max((evals_c ** 2).sum(), 1e-30)
    print(f"\n        effective rank (participation ratio): {part_ratio:.2f} on cosine, "
          f"{part_ratio_c:.2f} after removing mu   (isotropic would be {K} and {K-1})")

    # ---- 3. mu / residual split -------------------------------------------
    # Centre the UNIT vectors, not the raw ones. Centring raw vectors lets the
    # largest-norm accent dominate mu, after which every short vector's residual is
    # mostly "not that accent" and short vectors correlate with each other for a
    # reason that is purely magnitude. Verified on synthetic data with a known
    # family structure: raw centring buried the true pair and invented a false one.
    # Magnitude is reported separately in step 1; this step is about direction.
    concentration = np.sqrt(max(C.mean(), 0.0))
    print(f"\n[gram] alignment concentration ||mean unit tau|| = {concentration:.4f}"
          f"   (1.0 = all identical direction, {1/np.sqrt(K):.2f} = mutually unrelated)")

    Rc = cosine_from_gram(residual_gram(C))
    _write_matrix(out_dir / "residual_cosine.csv", names, Rc, fmt="{:.4f}")
    off = Rc[~np.eye(K, dtype=bool)]
    baseline = residual_cosine_baseline(K)
    print(f"        residual cosine: mean {off.mean():+.4f}  vs  "
          f"structural baseline {baseline:+.4f} (= -1/(K-1))")
    print("        ^ compare against the BASELINE, not against zero.")

    # ---- 4. leave-one-out reconstruction ----------------------------------
    print("\n[gram] leave-one-out reconstruction")
    print(f"        {'held out':<10} {'R2 full':>9} {'R2 mean':>9} {'increment':>10}   top coefficients")
    rows = []
    for k, name in enumerate(names):
        w, r2_full, r2_mean = loo_reconstruction(G, k)
        others = [n for n in names if n != name]
        order = np.argsort(-np.abs(w))
        top = ", ".join(f"{others[i]} {w[i]:+.3f}" for i in order[:3])
        inc = r2_full - r2_mean
        print(f"        {name:<10} {r2_full:9.4f} {r2_mean:9.4f} {inc:10.4f}   {top}")
        rows.append([name, f"{r2_full:.6f}", f"{r2_mean:.6f}", f"{inc:.6f}",
                     ";".join(f"{o}={c:.6f}" for o, c in zip(others, w))])
    with open(out_dir / "loo_reconstruction.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["held_out", "r2_full", "r2_mean_only", "increment", "coefficients"])
        wr.writerows(rows)

    increments = np.array([float(r[3]) for r in rows])

    # ---- 5. in-family vs singleton folds ----------------------------------
    # Convex combinations only reach the convex hull, so a fold whose accent has an
    # in-family neighbour is interpolation while a family singleton is extrapolation.
    # The ordering is predicted in advance, which makes it a real test.
    if families:
        fam = [families.get(n) for n in names]
        counts = {f: fam.count(f) for f in set(fam) if f is not None}
        paired = [i for i, f in enumerate(fam) if f is not None and counts[f] > 1]
        single = [i for i, f in enumerate(fam) if f is not None and counts[f] == 1]
        if paired and single:
            print(f"\n[gram] interpolation folds (in-family neighbour, n={len(paired)}): "
                  f"mean increment {increments[paired].mean():+.4f}")
            print(f"       extrapolation folds (family singleton,  n={len(single)}): "
                  f"mean increment {increments[single].mean():+.4f}")
            if increments[paired].mean() <= increments[single].mean():
                print("       ^ NOT in the predicted order: whatever structure exists "
                      "is not tracking language family.")

    # ---- 6. noise floor from within-accent replicates ----------------------
    # tau_k is itself a noisy draw. Asking for R^2 = 1 is asking for the wrong thing;
    # a prediction is only weak relative to how well the accent reproduces ITSELF.
    if replicate_paths:
        print("\n[gram] within-accent noise floor (same accent, different checkpoint/seed)")
        floor = []
        for name, path in replicate_paths.items():
            if name not in vectors:
                print(f"        {name}: no matching --vector, skipped")
                continue
            rep = lora_factors(load_lora_vector(path), rank, include, exclude)
            c = inner(vectors[name], rep) / np.sqrt(
                max(inner(vectors[name], vectors[name]) * inner(rep, rep), 1e-30))
            floor.append(c)
            print(f"        {name:<10} cos(tau, replicate) = {c:.4f}   <- {path}")
        if floor:
            print(f"        floor (min) = {min(floor):.4f} -- read every reconstruction "
                  "cosine against this, not against 1.0")

    # ---- verdict -----------------------------------------------------------
    print("\n[gram] verdict")
    if share[0] > 0.90:
        print("        COLLAPSED: one direction holds >90% of the aligned energy. Vectors")
        print("        are near-parallel -- a held-out vector is 'predictable' only via the")
        print("        mean, so accents cannot be told apart. Check the increment column.")
    elif share[0] < 1.5 / K:
        print("        ISOTROPIC: directions are near-mutually-orthogonal, so no vector")
        print("        informs any other. Reconstruction is hopeless whatever the")
        print("        coefficient method -- a valid, reportable negative result.")
    else:
        print(f"        STRUCTURED: leading direction holds {share[0]:.1%} of the aligned")
        print(f"        energy, effective rank {part_ratio:.2f} of a possible {K}.")
        print("        A shared component plus further real axes -- the shape RQ5 needs.")
    print(f"        median LOO increment over mean-only: {np.median(increments):+.4f}")
    print("        (the increment, not R2_full, is what H5a stands on)")
    print(f"\n[gram] wrote {out_dir}")

    return {"names": names, "gram": G, "increments": increments}


def main():
    p = argparse.ArgumentParser(description="RQ5 go/no-go: accent-vector Gram matrix")
    p.add_argument("--vector", action="append", default=[], required=True,
                   help="name=path to a LoRA accent vector / snapshot (repeatable)")
    p.add_argument("--replicate", action="append", default=[],
                   help="name=path to a second vector for the SAME accent (noise floor)")
    p.add_argument("--family", action="append", default=[],
                   help="name=family, e.g. hindi=indic (enables the interpolation "
                        "vs extrapolation comparison)")
    p.add_argument("--rank", type=int, default=16, help="LoRA rank used in training")
    p.add_argument("--include", action="append", default=[],
                   help="only keep LoRA keys containing this substring (repeatable)")
    p.add_argument("--exclude", action="append", default=[],
                   help="drop LoRA keys containing this substring, e.g. text_embed")
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()

    run(_pairs(a.vector), _pairs(a.replicate), _pairs(a.family), a.rank,
        a.include or None, a.exclude or None, a.out_dir)


if __name__ == "__main__":
    main()

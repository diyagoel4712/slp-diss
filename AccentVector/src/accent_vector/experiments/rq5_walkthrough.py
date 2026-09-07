"""RQ5 walkthrough -- the same objects as ``rq5_gram``, explained as they print.

``rq5_gram`` is the experiment; this is the tutorial. It computes nothing new -- it
imports the same functions -- but narrates each object so the abstractions land on
real numbers instead of on notation.

Run it with no vectors at all to see a worked example on synthetic accents with a
known family structure (so you know what a *positive* result looks like), then run
it again pointed at the real snapshots and compare:

    python -m accent_vector.experiments.rq5_walkthrough --demo

    python -m accent_vector.experiments.rq5_walkthrough \
        --vector british=<ckpt>/british/snapshots/lora_60000.pt \
        --vector dutch=<ckpt>/dutch/snapshots/lora_60000.pt \
        --vector hindi=<ckpt>/hindi/snapshots/lora_60000.pt \
        --vector bengali=<ckpt>/bengali/snapshots/lora_60000.pt

CPU-only; does not import f5_tts.
"""

import argparse
import textwrap

import numpy as np
import torch

from accent_vector.experiments.rq5_gram import (
    cosine_from_gram,
    gram,
    load_lora_vector,
    loo_reconstruction,
    lora_factors,
    residual_cosine_baseline,
    residual_gram,
    _pairs,
)

W = 82


def rule(title):
    print("\n" + "=" * W)
    print(f"  {title}")
    print("=" * W)


def say(text, indent="  "):
    print(textwrap.fill(textwrap.dedent(text).strip(), width=W,
                        initial_indent=indent, subsequent_indent=indent))
    print()


def bar(frac, width=44, ch="#"):
    n = int(round(max(frac, 0.0) * width))
    return ch * n + "." * (width - n)


def show_matrix(names, M, fmt="{:+.3f}"):
    pad = max(len(n) for n in names)
    print(" " * (pad + 2) + "".join(f"{n[:7]:>9}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:<{pad}}" + "".join(fmt.format(M[i, j]).rjust(9)
                                        for j in range(len(names))))
    print()


def demo_vectors(rank=16):
    """Synthetic accents with a deliberate structure: a component every accent
    shares, a component shared only within a family, and per-accent noise -- plus
    very unequal norms, because real fine-tunes are not norm-matched either.

    This is what a POSITIVE result looks like. Real vectors are compared to it.
    """
    g = torch.Generator().manual_seed(11)
    IN, OUT = 128, 96
    A = torch.randn(rank, IN, generator=g)          # shared encoder side
    shared = torch.randn(OUT, rank, generator=g)
    fam_vec = {f: torch.randn(OUT, rank, generator=g) for f in ("germanic", "indic")}
    # arabic/mandarin are family SINGLETONS -- labelled, but with no partner to
    # share a block with, which is what makes their folds extrapolation.
    spec = {"british": ("germanic", 1.0), "dutch": ("germanic", 0.4),
            "hindi": ("indic", 2.2), "bengali": ("indic", 0.7),
            "arabic": ("semitic", 1.4), "mandarin": ("sinitic", 0.6)}
    out, fams = {}, {}
    for name, (fam, scale) in spec.items():
        B = 1.0 * shared + torch.randn(OUT, rank, generator=g) * 0.8
        fams[name] = fam
        if fam in fam_vec:
            B = B + 1.3 * fam_vec[fam]
        out[name] = lora_factors({
            "blocks.0.attn.lora_q.encoders.weight": A.clone(),
            "blocks.0.attn.lora_q.decoders.weight": B * scale,
        }, rank)
    return out, fams


def run(vector_paths, families, rank, demo):
    if demo or not vector_paths:
        print("\n*** DEMO MODE: synthetic accents with a known family structure. ***")
        print("*** Re-run with --vector name=path to see your real numbers.     ***")
        vectors, fams = demo_vectors(rank)
        families = families or fams
    else:
        vectors = {}
        for name, path in vector_paths.items():
            vectors[name] = lora_factors(load_lora_vector(path), rank)

    names, G = gram(vectors)
    K = len(names)
    C = cosine_from_gram(G)
    norms = np.sqrt(np.diag(G))

    # ----------------------------------------------------------------- step 1
    rule("STEP 1  -- what an accent vector actually is")
    say("""
        Fine-tuning moved the model from theta_pre to theta_ft. The accent vector is
        just the difference: tau = theta_ft - theta_pre. It is one very long list of
        numbers -- one per parameter the fine-tune touched. Nothing more exotic than
        that. ||tau|| ("the norm") is its length: how far the fine-tune travelled.
        """)
    for n in names:
        n_modules = len(vectors[n])
        n_params = sum(B.numel() + A.numel() + (0 if b is None else b.numel())
                       for B, A, b in vectors[n].values())
        print(f"    {n:<10} {n_modules:>4} LoRA modules, {n_params:>10,} stored numbers, "
              f"||tau|| = {norms[names.index(n)]:.4g}")
    print()
    say("""
        Watch the norms. If one accent's vector is far longer than the others, that
        alone will dominate any comparison that does not divide the lengths out --
        which is exactly why step 3 uses cosine and not raw size.
        """)

    # ----------------------------------------------------------------- step 2
    rule("STEP 2  -- comparing two vectors: the inner product")
    say("""
        To ask "do these two fine-tunes push in similar directions?" you take the
        inner (dot) product: multiply the two vectors element by element and add it
        all up. Big positive = same direction. Zero = unrelated directions. Negative
        = opposing. Collecting every pair into a table gives the GRAM MATRIX,
        G[i][j] = <tau_i, tau_j>. That's the whole mystery: a table of dot products.
        (3Blue1Brown "Essence of Linear Algebra" Ch. 9 is the picture for this.)
        """)
    say("""
        The diagonal is each vector dotted with itself, i.e. its squared length. So
        the raw Gram mixes together two different things -- how LONG each vector is,
        and how ALIGNED they are. Dividing each entry by the two lengths removes the
        length and leaves pure alignment: the cosine.
        """)

    # ----------------------------------------------------------------- step 3
    rule("STEP 3  -- the cosine matrix (alignment, with size divided out)")
    say("""
        cos = +1 identical direction, 0 unrelated, -1 opposite. Read the off-diagonal
        entries: those are your accent-similarity numbers.
        """)
    show_matrix(names, C)
    iu = np.triu_indices(K, 1)
    off = C[iu]
    hi = int(np.argmax(off))
    lo = int(np.argmin(off))
    print(f"    most aligned pair : {names[iu[0][hi]]} / {names[iu[1][hi]]}  cos = {off[hi]:+.3f}")
    print(f"    least aligned pair: {names[iu[0][lo]]} / {names[iu[1][lo]]}  cos = {off[lo]:+.3f}")
    print(f"    mean off-diagonal cosine: {off.mean():+.3f}\n")
    if off.mean() > 0.5:
        say("""
            NOTE the mean is high. That usually means every fine-tune shares a big
            common direction -- plausibly "adapt off the pretraining distribution"
            rather than anything about accent. Step 5 separates that out.
            """)

    # ----------------------------------------------------------------- step 4
    rule("STEP 4  -- the eigenspectrum: how many directions are really here?")
    say("""
        Six vectors could in principle point six completely different ways, or they
        could all lie nearly along one line. The eigenvalues of the cosine matrix
        answer which. This is exactly PCA (StatQuest's PCA video is the same idea):
        each eigenvalue is how much of the spread lies along one independent
        direction. Because we used cosines, they sum to K, so an evenly-spread
        ("isotropic") set gives every eigenvalue a share of 1/K.
        """)
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    share = ev / ev.sum()
    for i in range(K):
        print(f"    direction {i+1}   share {share[i]:6.3f}   {bar(share[i])}")
    eff = (ev.sum() ** 2) / (ev ** 2).sum()
    print()
    print(f"    effective rank = {eff:.2f}   (isotropic would be {K:.2f}; "
          f"all-parallel would be 1.00)\n")
    say(f"""
        Both extremes are bad news for RQ5. If direction 1 held ~all the share, the
        accents are near-parallel: you could 'predict' a held-out one, but only by
        reproducing the average, so accents are indistinguishable. If all shares were
        1/{K} = {1/K:.3f}, the vectors are mutually unrelated and nothing predicts
        anything. What the research question needs is the middle: one big shared
        direction plus two or three further real ones.
        """)

    # ----------------------------------------------------------------- step 5
    rule("STEP 5  -- splitting off the shared component")
    say("""
        Write mu for the average DIRECTION of the vectors, and r_i for what is left of
        each accent once that shared direction is removed. The claim in H5a is that
        the residuals r_i, not the raw vectors, are what carry accent identity.
        """)
    say("""
        Note we average the vectors after setting them all to unit length. Averaging
        them raw would let the single longest vector dominate mu, and then every short
        vector's residual is mostly just "not that one" -- so short vectors would
        appear related to each other purely because they are short. Size was already
        reported in step 1; this step is only about direction.
        """)
    conc = np.sqrt(max(C.mean(), 0.0))
    print(f"    alignment concentration ||mean unit tau|| = {conc:.3f}    "
          f"(1.0 = one direction; {1/np.sqrt(K):.2f} = unrelated)\n")
    Rc = cosine_from_gram(residual_gram(C))
    show_matrix(names, Rc)
    base = residual_cosine_baseline(K)
    off_r = Rc[iu]
    print(f"    mean residual cosine = {off_r.mean():+.3f}")
    print(f"    structural baseline  = {base:+.3f}   (= -1/(K-1))\n")
    say(f"""
        IMPORTANT and easy to get wrong: subtracting a mean forces the residuals to
        sum to zero, which mechanically makes their average pairwise cosine negative
        -- about {base:+.3f} for K={K}. So a negative number here is NOT evidence that
        accents are dissimilar. Compare against the baseline, never against zero.
        A pair sitting clearly ABOVE {base:+.3f} is genuinely related.
        """)
    above = [(names[iu[0][t]], names[iu[1][t]], off_r[t])
             for t in np.argsort(-off_r)[:3]]
    for a, b, v in above:
        flag = "  <- above baseline" if v > base else ""
        print(f"    {a} / {b}: {v:+.3f}{flag}")
    print()

    # ----------------------------------------------------------------- step 6
    rule("STEP 6  -- one leave-one-out fold, narrated")
    # prefer a fold whose accent has an in-family partner: that is the interpolation case
    pick = 0
    if families:
        fam = [families.get(n) for n in names]
        counts = {f: fam.count(f) for f in set(fam) if f is not None}
        for i, f in enumerate(fam):
            if f is not None and counts[f] > 1:
                pick = i
                break
    held = names[pick]
    others = [n for n in names if n != held]
    say(f"""
        Hide '{held}' and try to rebuild its vector from the other {K-1}. That means
        choosing coefficients w so that the weighted sum of the remaining vectors
        lands as close as possible to the hidden one. "As close as possible" in the
        least-squares sense is a projection onto the subspace they span -- Strang's
        'projection onto subspaces' lecture is precisely this operation.
        """)
    w, r2_full, r2_mean = loo_reconstruction(G, pick)
    print(f"    coefficients that best rebuild '{held}':")
    for o, c in sorted(zip(others, w), key=lambda t: -abs(t[1])):
        marker = ""
        if families and families.get(o) and families.get(o) == families.get(held):
            marker = "   <- same language family"
        print(f"        {o:<10} {c:+.4f}{marker}")
    print()
    say("""
        Now: how good is that rebuild? R^2 is the fraction of the hidden vector the
        rebuild accounts for -- 1.0 is perfect, 0.0 is useless. But R^2 on its own is
        misleading here, because you can score well just by reproducing the shared
        component mu. So we also fit the one-knob model "tau ~ c * mu" and report the
        INCREMENT. Only the increment is evidence of accent-SPECIFIC structure.
        """)
    inc = r2_full - r2_mean
    print(f"    R^2 using all {K-1} vectors        : {r2_full:.4f}")
    print(f"    R^2 using only the average mu      : {r2_mean:.4f}")
    print(f"    increment (what the others add)    : {inc:+.4f}   <- THE number\n")
    if inc < 0.02:
        say("""
            That increment is essentially zero: the other accents add nothing beyond
            the average. If this holds across folds on the real vectors, RQ5's
            weight-space arm has a clear negative answer -- which is a reportable
            finding, not a failed experiment.
            """)
    elif inc < 0.15:
        say("""
            A small but non-zero increment: some accent-specific signal, but most of
            what is reconstructible is just the shared component. Report both numbers,
            never R^2 alone.
            """)
    else:
        say("""
            A substantial increment: the other accents carry real information about
            this one beyond the average. That is the result RQ5 needs, and the next
            question is whether typological distance predicts those coefficients.
            """)

    # ----------------------------------------------------------------- step 7
    rule("STEP 7  -- all folds, and what to do next")
    print(f"    {'held out':<12}{'R2 full':>10}{'R2 mean':>10}{'increment':>12}")
    incs = []
    for k, n in enumerate(names):
        _, a, b = loo_reconstruction(G, k)
        incs.append(a - b)
        print(f"    {n:<12}{a:10.4f}{b:10.4f}{a-b:12.4f}")
    incs = np.array(incs)
    print()
    if families:
        fam = [families.get(n) for n in names]
        counts = {f: fam.count(f) for f in set(fam) if f is not None}
        paired = [i for i, f in enumerate(fam) if f is not None and counts[f] > 1]
        single = [i for i, f in enumerate(fam) if f is not None and counts[f] == 1]
        if paired and single:
            say(f"""
                Accents WITH an in-family neighbour are an interpolation problem;
                family singletons are an extrapolation problem, and convex
                combinations only ever reach the region their neighbours surround.
                So the first group should score higher. It is a prediction made in
                advance, which is what makes it a test.
                """)
            print(f"    with in-family neighbour (n={len(paired)}): "
                  f"mean increment {incs[paired].mean():+.4f}")
            print(f"    family singleton         (n={len(single)}): "
                  f"mean increment {incs[single].mean():+.4f}\n")

    say(f"""
        Median increment across folds: {np.median(incs):+.4f}. That single number is
        the go/no-go for the composition experiments. Everything above came from one
        {K}x{K} table of dot products -- no synthesis, no GPU.
        """)
    say("""
        Next: run `rq5_gram` for the full set of CSVs, and re-run it with
        `--exclude text_embed --exclude input_embed` once Mandarin exists, since its
        vector carries mass in pinyin embedding rows no other accent touches.
        """)


def main():
    p = argparse.ArgumentParser(description="RQ5 walkthrough: the geometry, narrated")
    p.add_argument("--vector", action="append", default=[],
                   help="name=path to a LoRA accent vector (repeatable); omit for --demo")
    p.add_argument("--family", action="append", default=[], help="name=family")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--demo", action="store_true",
                   help="run on synthetic accents with a known family structure")
    a = p.parse_args()
    run(_pairs(a.vector), _pairs(a.family), a.rank, a.demo)


if __name__ == "__main__":
    main()

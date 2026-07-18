"""Helpers shared across two or more of the RQ experiment modules:

    load_eval                     eval-suite bridge      (rq1, rq2, rq3)
    alpha_dirs, wavs_in           synthesis-grid IO      (rq1, rq2, rq3)
    cosine_matrix, classical_mds  vector geometry        (rq2, viz_temporal)
    leakage_onset                 threshold crossing     (rq1, rq_temporal)

Single-use helpers live in the RQ module that uses them: the vector flattener and
Mantel test in rq2_geometry; the voicing-rhythm math and gap-closure in
rq3_decomposition; utt_index in rq1_reproduction.
"""

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO / "Evaluation"          # reusable metric suite (evaluation_functions.py)


# --- eval-suite bridge ------------------------------------------------------
def load_eval():
    """Import the repo's evaluation_functions (kept unchanged; the plan reuses it)."""
    if str(EVAL_DIR) not in sys.path:
        sys.path.insert(0, str(EVAL_DIR))
    import evaluation_functions as ef  # noqa: E402
    return ef


# --- synthesis-grid IO ------------------------------------------------------
def alpha_dirs(sweep_dir):
    """[(alpha, dir)] for alpha_<a> subdirs, sorted by alpha; falls back to the
    dir itself as a single (None, dir) set."""
    sweep_dir = Path(sweep_dir)
    found = []
    for d in (sweep_dir.iterdir() if sweep_dir.is_dir() else []):
        m = re.fullmatch(r"alpha_([0-9.]+)", d.name)
        if d.is_dir() and m:
            found.append((float(m.group(1)), d))
    return sorted(found, key=lambda t: t[0]) or [(None, sweep_dir)]


def wavs_in(d):
    return sorted(str(p) for p in Path(d).glob("*.wav"))


# --- vector geometry (RQ2 map + RQ6 trajectory viz) -------------------------
def cosine_matrix(vectors):
    """vectors: dict name -> 1-D array. Returns (names, NxN cosine matrix)."""
    names = list(vectors)
    mats = [np.asarray(vectors[n], dtype=float) for n in names]
    norms = [m / (np.linalg.norm(m) + 1e-12) for m in mats]
    n = len(names)
    C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            C[i, j] = C[j, i] = float(norms[i] @ norms[j])
    return names, C


def classical_mds(sim, n_components=2):
    """Classical (Torgerson) MDS from a similarity matrix in [-1, 1], via the
    distance D = sqrt(2(1 - sim)). Returns an (N, n_components) embedding.
    Dependency-light stand-in for sklearn.manifold.MDS."""
    sim = np.asarray(sim, dtype=float)
    D2 = 2.0 * (1.0 - sim)                       # squared distances
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J                        # double-centred Gram
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order][:n_components], vecs[:, order][:, :n_components]
    return vecs * np.sqrt(np.clip(vals, 0, None))


# --- threshold crossing over a swept series (RQ1b onset, RQ6 convergence) ----
def leakage_onset(alphas, signal, threshold, rising=True):
    """Linearly-interpolated alpha at which a signal first crosses a threshold.

    Reused for two "where does it cross" questions: RQ1b language-leakage onset
    (how far the vector scales before content leaves the base language /
    intelligibility collapses) and RQ6 direction-convergence step.

    rising=True  : onset where the signal climbs past threshold (e.g. WER).
    rising=False : onset where the signal drops below threshold (e.g. P(English)).
    Returns the crossing alpha, the lowest alpha if already past it there, or nan
    if it never crosses. None alphas and nan signal values are dropped.
    """
    pts = sorted((a, s) for a, s in zip(alphas, signal)
                 if a is not None and s is not None and not np.isnan(s))
    if not pts:
        return float("nan")
    first_a, first_s = pts[0]
    if (rising and first_s > threshold) or (not rising and first_s < threshold):
        return float(first_a)  # already past it at the baseline
    for (a0, s0), (a1, s1) in zip(pts, pts[1:]):
        if (rising and s0 <= threshold < s1) or (not rising and s0 >= threshold > s1):
            frac = 0.0 if s1 == s0 else (threshold - s0) / (s1 - s0)
            return float(a0 + frac * (a1 - a0))
    return float("nan")

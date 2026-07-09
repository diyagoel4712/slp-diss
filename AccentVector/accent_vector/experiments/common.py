"""Shared utilities for the experiment harness: grid IO, vector geometry,
rhythm math, gap-closure, and the eval-suite bridge.

The pure-math helpers (cosine_matrix, mantel_test, classical_mds, npvi,
gap_closure_*) have no heavy dependencies beyond numpy and are unit-tested in
tests_math.py.
"""

import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SOTA = REPO / "SOTA_models_experiments"


# --- eval-suite bridge ------------------------------------------------------
def load_eval():
    """Import the repo's evaluation_functions (kept unchanged; the plan reuses it)."""
    if str(SOTA) not in sys.path:
        sys.path.insert(0, str(SOTA))
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


def utt_index(path):
    """Recover the transcript index from a utt####.wav name (or None)."""
    m = re.search(r"utt(\d+)", Path(path).stem)
    return int(m.group(1)) if m else None


# --- accent-vector geometry (RQ2 / RQ3.4) -----------------------------------
def load_vector_flat(vector_path, include=None, exclude=None):
    """Flatten a saved accent vector into one 1-D numpy array over its float
    tensors, in sorted-key order (stable across accents so vectors are
    comparable). Optional include/exclude substrings restrict to a layer subset.

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


# --- rhythm / suprasegmental math (RQ3.2) -----------------------------------
def run_lengths(mask):
    """Lengths of consecutive-True runs in a boolean array (e.g. voiced runs)."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    idx = np.flatnonzero(np.diff(mask.astype(int)))
    bounds = np.concatenate(([0], idx + 1, [mask.size]))
    return [int(bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)
            if mask[bounds[i]]]


def npvi(durations):
    """normalized Pairwise Variability Index over successive interval durations
    (a standard speech-rhythm descriptor). Higher = more stress-timed."""
    d = np.asarray(durations, dtype=float)
    if d.size < 2:
        return float("nan")
    num = np.abs(d[:-1] - d[1:])
    den = (d[:-1] + d[1:]) / 2.0
    ok = den > 0
    return float(100.0 / (d.size - 1) * np.sum(num[ok] / den[ok]))


def voicing_rhythm(f0, hop_s):
    """Alignment-free suprasegmental descriptors from an F0 track (extract_f0):
    voiced fraction (%V proxy), nPVI over voiced-run durations, and articulation
    rate (voiced runs per second). A voicing-based proxy for %V/nPVI that needs
    no forced aligner; swap in MFA vowel/consonant intervals for the rigorous
    version."""
    f0 = np.asarray(f0, dtype=float)
    voiced = ~np.isnan(f0)
    total = voiced.size
    runs = run_lengths(voiced)
    dur_s = [r * hop_s for r in runs]
    duration_s = total * hop_s if total else 0.0
    return {
        "pct_voiced": float(voiced.mean()) if total else float("nan"),
        "npvi_voiced": npvi(dur_s),
        "artic_rate": (len(runs) / duration_s) if duration_s else float("nan"),
        "f0_mean": float(np.nanmean(f0)) if voiced.any() else float("nan"),
        "f0_std": float(np.nanstd(f0)) if voiced.any() else float("nan"),
        "f0_range": (float(np.nanmax(f0) - np.nanmin(f0)) if voiced.any() else float("nan")),
    }


# --- gap-closure (RQ3.3) ----------------------------------------------------
def gap_closure_scalar(base, alpha_val, natural):
    """Fraction of the baseline->natural gap a scalar feature closes at this
    alpha: (x_alpha - x_base) / (x_natural - x_base). ~1 => moved fully to the
    natural-accent value; ~0 => did not move."""
    denom = natural - base
    if abs(denom) < 1e-12:
        return float("nan")
    return float((alpha_val - base) / denom)


def gap_closure_distance(d_base_to_natural, d_alpha_to_natural):
    """Fraction of a distance-to-natural closed (e.g. PPG-KL): 1 - d_alpha/d_base."""
    if abs(d_base_to_natural) < 1e-12:
        return float("nan")
    return float(1.0 - d_alpha_to_natural / d_base_to_natural)


# --- language leakage onset (RQ1b) ------------------------------------------
def leakage_onset(alphas, signal, threshold, rising=True):
    """Linearly-interpolated alpha at which a leakage signal first crosses a
    threshold -- how far the vector can be scaled before content leaves the base
    language / intelligibility collapses (RQ1b).

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
        return float(first_a)  # already leaking at the baseline
    for (a0, s0), (a1, s1) in zip(pts, pts[1:]):
        if (rising and s0 <= threshold < s1) or (not rising and s0 >= threshold > s1):
            frac = 0.0 if s1 == s0 else (threshold - s0) / (s1 - s0)
            return float(a0 + frac * (a1 - a0))
    return float("nan")


# --- speaker metadata (RQ5.1 gender split) ----------------------------------
def vctk_gender_map(vctk_root):
    """speaker_id -> 'M'/'F' from VCTK speaker-info.txt."""
    info = Path(vctk_root) / "speaker-info.txt"
    out = {}
    if not info.exists():
        return out
    for i, line in enumerate(info.read_text(encoding="utf-8").splitlines()):
        if i == 0:
            continue
        p = line.split()
        if len(p) >= 3:
            out[p[0]] = p[2]
    return out


def speaker_from_wav(path):
    """Best-effort speaker id from a filename like p225_003 or ABA_arctic_a0001."""
    stem = Path(path).stem
    return stem.split("_")[0]

"""Direct test of whether Parler's accent descriptor changes anything.

Parler's output depends on (accent, gender, utterance) -- the ONLY thing that varies across
the four accents for a fixed gender+utterance is the word "{accent}" in the style prompt. So
we embed those four versions in GenAID's speaker/content-agnostic accent space and measure
their pairwise cosine similarity. If the accent word is ignored, the four embeddings are
near-identical (cross-accent similarity ~1.0); a model that actually renders the accent would
push them apart (lower similarity).

Run with the main evaluation interpreter; it shells out to the GenAID environment for the
accent embeddings (override with GENAID_PYTHON, as in run_eval.py).
"""
from itertools import combinations
from pathlib import Path

import numpy as np

import eval_config as cfg
import evaluation_functions as ef


def first_speaker(accent, gender):
    for s in cfg.ACCENT_SPEAKERS[accent]:
        if cfg.SPEAKER_GENDER[s] == gender:
            return s
    return None


def main():
    accents = list(cfg.ACCENT_SPEAKERS)
    groups, paths = [], []
    for gender in ("M", "F"):
        for utt in cfg.UTTERANCES:
            row = {}
            for a in accents:
                spk = first_speaker(a, gender)
                if spk is None:
                    continue
                p = str(cfg.synth_path("parler", a, spk, utt))
                if Path(p).exists():
                    row[a] = p
                    paths.append(p)
            if len(row) >= 2:
                groups.append((gender, utt, row))

    if not paths:
        raise SystemExit("no Parler clips found -- has parler synthesis run?")

    preds = ef.predict_accent_genaid(paths, with_embeddings=True)
    emb = {d["wav"]: np.asarray(d["embedding"], float) for d in preds}

    def lookup(p):
        return emb.get(p, emb.get(str(Path(p).resolve())))

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    sims = []
    for _gender, _utt, row in groups:
        vecs = {a: lookup(p) for a, p in row.items()}
        for a1, a2 in combinations(row, 2):
            if vecs[a1] is not None and vecs[a2] is not None:
                sims.append(cos(vecs[a1], vecs[a2]))

    sims = np.array(sims)
    print("Parler cross-accent accent-embedding cosine similarity (same gender+utterance):")
    print(f"  pairs={len(sims)}  mean={sims.mean():.4f}  median={np.median(sims):.4f}"
          f"  min={sims.min():.4f}  max={sims.max():.4f}")
    print("  (≈1.0 => the accent word is ignored; the four 'accents' are the same voice.)")
    return float(sims.mean())


if __name__ == "__main__":
    main()

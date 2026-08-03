#!/usr/bin/env python3
"""Extract ECAPA-TDNN speaker embeddings for speaker-similarity evaluation.

Loads speechbrain/spkrec-ecapa-voxceleb (the standard VoxCeleb speaker encoder) and
emits a 192-d speaker embedding per wav as a JSON list (same shape as the accent
wrappers). Cosine similarity between paired synth/reference embeddings is computed
downstream by evaluation_functions.speaker_similarity.

Usage
-----
python predict_speaker_embeddings.py --wav_list wavs.txt --out emb.json [--device cpu]
"""
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_list", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--source", default="speechbrain/spkrec-ecapa-voxceleb")
    ap.add_argument("--savedir", default="./spkrec_ecapa")
    args = ap.parse_args()

    from speechbrain.pretrained import EncoderClassifier
    import librosa
    import torch

    classifier = EncoderClassifier.from_hparams(
        source=args.source,
        savedir=args.savedir,
        run_opts={"device": args.device},
    )

    with open(args.wav_list) as f:
        wavs = [ln.strip() for ln in f if ln.strip()]

    results = []
    for w in wavs:
        # load via librosa to avoid torchaudio's torchcodec backend dependency.
        sig, _ = librosa.load(w, sr=16000)
        wav = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)  # (1, T)
        emb = classifier.encode_batch(wav).squeeze().tolist()      # (192,)
        results.append({"wav": w, "embedding": emb})
        print(f"[ECAPA] {os.path.basename(w)} -> {len(emb)}-d embedding", file=sys.stderr)

    out = json.dumps(results, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Secondary accent-ID via SpeechBrain CommonAccent (ECAPA), as a sanity check
against GenAID so accent conclusions don't hinge on a single model.

Loads Jzuluaga/accent-id-commonaccent_ecapa from the HuggingFace Hub and emits
the predicted accent label + posteriors per wav, as a JSON list (same shape as
predict_GenAID.py, minus embeddings).

Usage
-----
python predict_commonaccent.py --wav_list wavs.txt --out preds.json [--device cpu]
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
    ap.add_argument("--source", default="Jzuluaga/accent-id-commonaccent_ecapa")
    ap.add_argument("--savedir", default="./commonaccent_ecapa")
    args = ap.parse_args()

    from speechbrain.pretrained import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(
        source=args.source,
        savedir=args.savedir,
        run_opts={"device": args.device},
    )
    # label set lives on the loaded label encoder
    ind2lab = classifier.hparams.label_encoder.ind2lab

    with open(args.wav_list) as f:
        wavs = [ln.strip() for ln in f if ln.strip()]

    import librosa
    import torch

    results = []
    for w in wavs:
        # load via librosa to avoid torchaudio's torchcodec backend dependency,
        # then classify the waveform tensor directly.
        sig, _ = librosa.load(w, sr=16000)
        wav = torch.tensor(sig, dtype=torch.float32).unsqueeze(0)  # (1, T)
        out_prob, score, index, text_lab = classifier.classify_batch(wav)
        probs = out_prob.squeeze(0).tolist()
        posteriors = {ind2lab[i]: float(probs[i]) for i in range(len(probs))}
        pred = text_lab[0] if isinstance(text_lab, list) else text_lab
        results.append({"wav": w, "pred_accent": pred, "posteriors": posteriors})
        print(f"[CommonAccent] {os.path.basename(w)} -> {pred}", file=sys.stderr)

    out = json.dumps(results, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()

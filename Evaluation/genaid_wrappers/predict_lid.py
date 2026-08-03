#!/usr/bin/env python3
"""Spoken language-ID via SpeechBrain VoxLingua107 (ECAPA), for RQ1b language
leakage: P(English) per clip, the direct "has content drifted out of English"
signal that WER conflates with the ASR's accent penalty.

Loads speechbrain/lang-id-voxlingua107-ecapa from the HuggingFace Hub and emits,
per wav, the posterior mass on English plus the top predicted language, as a JSON
list (same shape/plumbing as predict_commonaccent.py).

VoxLingua107 labels are ``"<iso>: <Language>"`` (e.g. ``"en: English"``); English
is located by its ISO code ``en`` so it is robust to label formatting.

Usage
-----
python predict_lid.py --wav_list wavs.txt --out preds.json [--device cpu]
"""
import argparse
import json
import os
import sys


def _english_index(ind2lab):
    """Index of the English class in VoxLingua107's label encoder, matched on the
    ISO code (``en``) rather than the display string."""
    for i, lab in ind2lab.items():
        code = str(lab).split(":", 1)[0].strip().lower()
        if code == "en" or str(lab).strip().lower() == "english":
            return i
    raise SystemExit(f"no English class found in label set: {list(ind2lab.values())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_list", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--source", default="speechbrain/lang-id-voxlingua107-ecapa")
    ap.add_argument("--savedir", default="./lid_voxlingua107")
    args = ap.parse_args()

    from speechbrain.pretrained import EncoderClassifier

    classifier = EncoderClassifier.from_hparams(
        source=args.source,
        savedir=args.savedir,
        run_opts={"device": args.device},
    )
    ind2lab = classifier.hparams.label_encoder.ind2lab
    en_idx = _english_index(ind2lab)

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
        # VoxLingua107's classify_batch returns log-probabilities (log-softmax);
        # exp() back to a normalised [0,1] distribution so p_english is a real
        # probability the rq1 leakage-onset threshold (P(English)<0.5) can use.
        raw = out_prob.squeeze(0)
        if float(raw.max()) <= 0.0:
            raw = raw.exp()
        probs = raw.tolist()
        p_english = float(probs[en_idx])
        pred = text_lab[0] if isinstance(text_lab, list) else text_lab
        results.append({"wav": w, "p_english": p_english,
                        "pred_lang": pred, "p_pred": float(max(probs))})
        print(f"[VoxLingua107] {os.path.basename(w)} -> {pred} "
              f"(P(en)={p_english:.3f})", file=sys.stderr)

    out = json.dumps(results, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()

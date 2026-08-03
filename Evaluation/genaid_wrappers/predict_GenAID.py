#!/usr/bin/env python3
"""Standalone GenAID inference for arbitrary wav files.

Loads the trained GenAID checkpoint and, for each input wav, emits the predicted
accent label, the full class posteriors, and the pooled accent embedding (the
statpool output, encoder_dim-d, before the classification MLP).

Designed to be called as a subprocess from the evaluation harness, which lives in
a different environment. Reads wav paths (one per line) from a file and writes a
JSON list to stdout (or --out).

Usage
-----
python predict_GenAID.py --wav_list wavs.txt \
    --pretrained_path ./GenAID_v6/save/CKPT+2024-07-21+23-06-07+00 \
    --wav2vec2_save ./pretrained_xlsr_large \
    --device cpu --out preds.json

Author: wrapper around inference_GenAID.py (Zuluaga 2023 / Zhong 2024).
"""
import argparse
import json
import os
import sys

import librosa
import torch
from hyperpyyaml import load_hyperpyyaml

SAMPLE_RATE = 16000
ENCODER_DIM = 1024
N_ACCENTS = 13
N_SPEAKERS = 11486  # size of the (unused) speaker-adversarial head, needed only so
                    # the checkpoint's ModuleList loads positionally without warnings


def build_model(pretrained_path, wav2vec2_save, device):
    """Construct GenAID modules from the v6 hparams and load the checkpoint."""
    # Minimal hparams mirroring inference_GenAID_v6.yaml, with portable paths.
    yaml = f"""
wav2vec2_hub: facebook/wav2vec2-large-xlsr-53
encoder_dim: {ENCODER_DIM}
n_accents: {N_ACCENTS}
activation: !name:torch.nn.GELU
dnn_layers: 2
dnn_neurons: 64

wav2vec2: !new:speechbrain.lobes.models.huggingface_wav2vec.HuggingFaceWav2Vec2
    source: !ref <wav2vec2_hub>
    output_norm: True
    freeze: True
    freeze_feature_extractor: True
    save_path: {wav2vec2_save}

avg_pool: !new:speechbrain.nnet.pooling.StatisticsPooling
    return_std: False

preout_mlp: !new:speechbrain.lobes.models.VanillaNN.VanillaNN
    input_shape: [null, null, !ref <encoder_dim>]
    activation: !ref <activation>
    dnn_blocks: !ref <dnn_layers>
    dnn_neurons: 64

output_mlp: !new:speechbrain.nnet.linear.Linear
    input_size: 64
    n_neurons: !ref <n_accents>
    bias: False

# speaker-adversarial head (GRL); unused at inference but kept so the checkpoint
# ModuleList loads positionally (preout=0, accent=1, adversarial=2).
output_mlp_adv: !new:speechbrain.nnet.linear.Linear
    input_size: 64
    n_neurons: {N_SPEAKERS}
    bias: False

log_softmax: !new:speechbrain.nnet.activations.Softmax
    apply_log: True

model: !new:torch.nn.ModuleList
    - [!ref <preout_mlp>, !ref <output_mlp>, !ref <output_mlp_adv>]

label_encoder: !new:speechbrain.dataio.encoder.CategoricalEncoder

pretrainer: !new:speechbrain.utils.parameter_transfer.Pretrainer
    loadables:
        model: !ref <model>
        wav2vec2: !ref <wav2vec2>
        label_encoder: !ref <label_encoder>
    paths:
        model: {pretrained_path}/model.ckpt
        wav2vec2: {pretrained_path}/wav2vec2.ckpt
        label_encoder: {pretrained_path}/../accent_encoder.txt
"""
    hparams = load_hyperpyyaml(yaml)

    # Load the label encoder mapping (index <-> accent label).
    accent_encoder = hparams["label_encoder"]
    accent_encoder.load_or_create(
        path=hparams["pretrainer"].paths["label_encoder"], output_key="accent"
    )

    # Pull the checkpoint weights into the modules. The checkpoint was saved on
    # CUDA, so pass device to set map_location (lets it load on a CPU-only machine).
    hparams["pretrainer"].collect_files()
    try:
        hparams["pretrainer"].load_collected(device=device)
    except TypeError:
        # older/newer signatures may not accept device kwarg
        hparams["pretrainer"].load_collected()

    for m in (hparams["wav2vec2"], hparams["preout_mlp"], hparams["output_mlp"]):
        m.to(device).eval()

    return hparams, accent_encoder


@torch.no_grad()
def predict_one(wav_path, hparams, accent_encoder, device):
    """Returns (pred_label, posteriors_dict, embedding_list) for one wav."""
    sig, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    sig = torch.tensor(sig, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T)

    feats = hparams["wav2vec2"](sig)                       # (1, T', encoder_dim)
    embedding = hparams["avg_pool"](feats).view(1, -1)     # (1, encoder_dim) = accent embedding
    hidden = hparams["preout_mlp"](embedding)
    logits = hparams["output_mlp"](hidden)
    log_probs = hparams["log_softmax"](logits).view(-1)    # (n_accents,)

    idx = int(torch.argmax(log_probs).item())
    pred_label = accent_encoder.ind2lab[idx]
    posteriors = {
        accent_encoder.ind2lab[i]: float(torch.exp(log_probs[i]))
        for i in range(len(log_probs))
    }
    return pred_label, posteriors, embedding.view(-1).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_list", required=True, help="file with one wav path per line")
    ap.add_argument("--pretrained_path",
                    default="./GenAID_v6/save/CKPT+2024-07-21+23-06-07+00")
    ap.add_argument("--wav2vec2_save", default="./pretrained_xlsr_large")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="output JSON path (default stdout)")
    ap.add_argument("--no_embedding", action="store_true",
                    help="omit embeddings from output to keep it small")
    args = ap.parse_args()

    with open(args.wav_list) as f:
        wavs = [ln.strip() for ln in f if ln.strip()]

    hparams, accent_encoder = build_model(
        args.pretrained_path, args.wav2vec2_save, args.device
    )

    results = []
    for w in wavs:
        pred, post, emb = predict_one(w, hparams, accent_encoder, args.device)
        row = {"wav": w, "pred_accent": pred, "posteriors": post}
        if not args.no_embedding:
            row["embedding"] = emb
        results.append(row)
        print(f"[GenAID] {os.path.basename(w)} -> {pred}", file=sys.stderr)

    out = json.dumps(results, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(out)
    else:
        print(out)


if __name__ == "__main__":
    main()

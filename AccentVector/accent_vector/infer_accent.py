"""Accent-controlled inference with F5-TTS (paper Section 3.4 + Figure 3).

For each strength coefficient alpha, we build theta_pre + alpha * tau_accent and
synthesize a fixed set of English transcripts. The reference clip is held
**fixed and neutral** across all alpha, so the accent vector is the only thing
that varies -- this isolates "accent from the vector" and makes the alpha sweep
interpretable (F5 clones the reference's own accent, so a varying reference
would confound the sweep; see ADAPTATION_PLAN.md gotcha #3).

Note on the paper deviation: XTTS pins a language-ID token to keep content
English while the delta supplies the accent. F5-TTS has no language-ID token,
so content language is set purely by the ``gen_text`` we feed. Feeding English
transcripts keeps content English while the merged vector shifts the acoustics.

Modes
-----
    # alpha sweep over a single accent vector (Eq. 4)
    python -m accent_vector.infer_accent \
        --pretrained ckpts/F5TTS_v1_Base/model_1250000.pt \
        --vector vectors/british.pt \
        --alphas 0,0.2,0.4,0.6,0.8,1.0 \
        --ref-audio refs/neutral.wav --ref-text "..." \
        --transcripts transcripts/eval_transcripts.txt \
        --out-dir results/british

    # synthesize a single, already-composed checkpoint (e.g. a mixed accent)
    python -m accent_vector.infer_accent \
        --ckpt ckpts/mixed/spanish+british.pt \
        --ref-audio refs/neutral.wav --ref-text "..." \
        --transcripts transcripts/eval_transcripts.txt \
        --out-dir results/spanish+british
"""

import argparse
import os
import tempfile
from importlib.resources import files

import soundfile as sf
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)

from accent_vector.extract_vector import compose


def load_transcripts(path):
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def build_model(config_path, ckpt_path, vocab_file, device):
    model_cfg = OmegaConf.load(config_path).model
    model_cls = get_class(f"f5_tts.model.{model_cfg.backbone}")
    mel_spec_type = model_cfg.mel_spec.mel_spec_type
    model = load_model(
        model_cls, model_cfg.arch, ckpt_path,
        mel_spec_type=mel_spec_type, vocab_file=vocab_file, device=device,
    )
    return model, mel_spec_type


def synthesize_set(model, vocoder, mel_spec_type, ref_audio, ref_text,
                   transcripts, out_dir, nfe_step, seed, device):
    os.makedirs(out_dir, exist_ok=True)
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
    for idx, gen_text in enumerate(transcripts):
        wave, sr, _ = infer_process(
            ref_audio, ref_text, gen_text, model, vocoder,
            mel_spec_type=mel_spec_type, nfe_step=nfe_step, seed=seed, device=device,
        )
        sf.write(os.path.join(out_dir, f"utt{idx:04d}.wav"), wave, sr)
    print(f"[infer] wrote {len(transcripts)} clips -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Accent-controlled F5-TTS inference")
    parser.add_argument("--config", default=None,
                        help="F5 model config yaml (default: packaged F5TTS_v1_Base.yaml)")
    parser.add_argument("--vocab", default="", help="vocab.txt (default: packaged pretrained vocab)")
    parser.add_argument("--ref-audio", required=True, help="fixed neutral reference clip")
    parser.add_argument("--ref-text", required=True, help="transcript of the reference clip")
    parser.add_argument("--transcripts", required=True, help="English transcripts, one per line")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")

    # sweep mode
    parser.add_argument("--pretrained", help="base checkpoint (theta_pre) for the sweep")
    parser.add_argument("--vector", help="accent vector for the sweep")
    parser.add_argument("--alphas", help="comma-separated strengths, e.g. 0,0.2,0.4,0.6,0.8,1.0")
    # single-checkpoint mode
    parser.add_argument("--ckpt", help="synthesize this pre-composed checkpoint directly")

    args = parser.parse_args()

    config_path = args.config or str(
        files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")
    )
    transcripts = load_transcripts(args.transcripts)
    vocoder = load_vocoder(vocoder_name="vocos")

    if args.ckpt:
        model, mel_spec_type = build_model(config_path, args.ckpt, args.vocab, args.device)
        synthesize_set(
            model, vocoder, mel_spec_type, args.ref_audio, args.ref_text,
            transcripts, args.out_dir, args.nfe, args.seed, args.device,
        )
        return

    if not (args.pretrained and args.vector and args.alphas):
        raise SystemExit("sweep mode needs --pretrained, --vector and --alphas (or use --ckpt)")

    alphas = [float(a) for a in args.alphas.split(",")]
    with tempfile.TemporaryDirectory() as tmp:
        for alpha in alphas:
            ckpt = os.path.join(tmp, f"accent_a{alpha}.pt")
            compose(args.pretrained, [(args.vector, alpha)], ckpt, verbose=False)
            model, mel_spec_type = build_model(config_path, ckpt, args.vocab, args.device)
            synthesize_set(
                model, vocoder, mel_spec_type, args.ref_audio, args.ref_text,
                transcripts, os.path.join(args.out_dir, f"alpha_{alpha}"),
                args.nfe, args.seed, args.device,
            )
            os.remove(ckpt)


if __name__ == "__main__":
    main()

"""Accent-controlled inference with F5-TTS (paper Section 3.4 + Figure 3).

For each strength coefficient alpha, we build theta_pre + alpha * tau_accent and
synthesize a fixed set of English transcripts (input text to be synthesised). The
reference clip is the target accent's **native-language (L1) reference**, held fixed
per accent across the sweep, so the accent vector is the only thing that varies
within a sweep (paper-faithful cloning; see ADAPTATION_PLAN.md gotcha #3). The sweep
runs between two exact anchors: **alpha=0 = theta_pre** (the pretrained model cloning
the accent from the reference alone, no fine-tuning) and **alpha=1 = theta_ft** (the
fully fine-tuned model, full accent-vector impact).

Note on the paper deviation: XTTS pins a language-ID token to keep content
English while the delta supplies the accent. F5-TTS has no language-ID token,
so content language is set purely by the ``gen_text`` we feed. Feeding English
transcripts keeps content English while the merged vector shifts the acoustics.

Two accent-vector tracks feed this (see AccentVector/README.md deviation #2):

* **LoRA (paper-matching, --lora).** The accent vector is the LoRA branch; the
  sweep builds the model ONCE and rescales the branch in place per alpha via
  ``lora_model.set_lora_alpha`` (no merge, exact ``theta_pre + alpha*theta_LoRA``).
* **Full fine-tune (merged checkpoint).** ``extract_vector.compose`` merges
  ``theta_pre + alpha*tau`` into a full checkpoint per alpha, which is then loaded.

Modes
-----
    # LoRA alpha sweep -- native, no merge (Eq. 3-4)
    python -m accent_vector.infer_accent --lora \
        --pretrained ckpts/F5TTS_v1_Base/model_1250000.pt \
        --lora-vector vectors/british_lora.pt \
        --config exps/.../config.yaml --vocab exps/.../vocab.txt \
        --alphas 0,0.2,0.4,0.6,0.8,1.0 \
        --ref-audio refs/england.wav --ref-text "..." \
        --transcripts transcripts/eval_transcripts.txt \
        --out-dir results/british \
        [--lora-label british --lora-mapping exps/.../lora_mapping.json]

    # full-fine-tune alpha sweep over a merged checkpoint diff (Eq. 4)
    python -m accent_vector.infer_accent \
        --pretrained ckpts/F5TTS_v1_Base/model_1250000.pt \
        --vector vectors/british.pt \
        --alphas 0,0.2,0.4,0.6,0.8,1.0 \
        --ref-audio refs/england.wav --ref-text "..." \
        --transcripts transcripts/eval_transcripts.txt \
        --out-dir results/british

    # synthesize a single, already-composed checkpoint (e.g. a mixed accent)
    python -m accent_vector.infer_accent \
        --ckpt ckpts/mixed/spanish+british.pt \
        --ref-audio refs/england.wav --ref-text "..." \
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
                   transcripts, out_dir, nfe_step, seed, device, lora_idx=None):
    os.makedirs(out_dir, exist_ok=True)
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
    for idx, gen_text in enumerate(transcripts):
        wave, sr, _ = infer_process(
            ref_audio, ref_text, gen_text, model, vocoder,
            mel_spec_type=mel_spec_type, nfe_step=nfe_step, seed=seed, device=device,
            lora_idx=lora_idx,
        )
        sf.write(os.path.join(out_dir, f"utt{idx:04d}.wav"), wave, sr)
    print(f"[infer] wrote {len(transcripts)} clips -> {out_dir}")


def synthesize_lora_sweep(pretrained, lora_vector, config_path, vocab, alphas,
                          ref_audio, ref_text, transcripts, out_dir,
                          nfe, seed, device, lora_idx=0):
    """Native LoRA alpha sweep: build the base+LoRA model once, then rescale the
    accent branch to each alpha in place (no per-alpha checkpoint merge). Writes
    ``<out_dir>/alpha_<a>/utt####.wav`` -- the same layout the merged path uses,
    so every downstream analysis reads it identically."""
    from accent_vector.lora_model import (
        build_base_model, load_lora_state, overlay_lora, set_lora_alpha,
    )

    model, cfg, vocoder = build_base_model(config_path, vocab, pretrained, device)
    mel_spec_type = cfg.model.mel_spec.mel_spec_type
    overlay_lora(model, load_lora_state(lora_vector, device))
    model.eval()
    for alpha in alphas:
        n = set_lora_alpha(model, alpha)
        print(f"[infer:lora] alpha={alpha} on {n} LoRA submodules (lora_idx={lora_idx})")
        synthesize_set(
            model, vocoder, mel_spec_type, ref_audio, ref_text,
            transcripts, os.path.join(out_dir, f"alpha_{alpha}"),
            nfe, seed, device, lora_idx=lora_idx,
        )


def main():
    parser = argparse.ArgumentParser(description="Accent-controlled F5-TTS inference")
    parser.add_argument("--config", default=None,
                        help="F5 model config yaml (default: packaged F5TTS_v1_Base.yaml)")
    parser.add_argument("--vocab", default="", help="vocab.txt (default: packaged pretrained vocab)")
    parser.add_argument("--ref-audio", required=True,
                        help="native-language (L1) reference clip, fixed across the sweep")
    parser.add_argument("--ref-text", required=True, help="transcript of the reference clip")
    parser.add_argument("--transcripts", required=True, help="English transcripts, one per line")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")

    # full-fine-tune sweep mode (merged checkpoint per alpha)
    parser.add_argument("--pretrained", help="base checkpoint (theta_pre) for the sweep")
    parser.add_argument("--vector", help="full-weight accent vector for the merged sweep")
    parser.add_argument("--alphas", help="comma-separated strengths, e.g. 0,0.2,0.4,0.6,0.8,1.0")
    # single-checkpoint mode
    parser.add_argument("--ckpt", help="synthesize this pre-composed checkpoint directly")
    # native-LoRA sweep mode (accent vector = LoRA branch; no merge)
    parser.add_argument("--lora", action="store_true",
                        help="native LoRA sweep: scale the LoRA branch by alpha in place")
    parser.add_argument("--lora-vector", help="LoRA accent vector / snapshot (lora_state_dict)")
    parser.add_argument("--lora-idx", type=int, default=None,
                        help="LoRA branch index (default: resolve via --lora-label/--lora-mapping, else 0)")
    parser.add_argument("--lora-label", help="accent label to look up in --lora-mapping")
    parser.add_argument("--lora-mapping", help="lora_mapping.json (label -> branch idx)")

    args = parser.parse_args()

    transcripts = load_transcripts(args.transcripts)

    # --- native LoRA sweep: build once, rescale the branch per alpha ---
    if args.lora:
        from accent_vector.lora_model import resolve_lora_idx
        if not (args.pretrained and args.lora_vector and args.alphas):
            raise SystemExit("--lora sweep needs --pretrained, --lora-vector and --alphas")
        if not (args.config and args.vocab):
            raise SystemExit("--lora sweep needs --config and --vocab from the training run")
        lora_idx = args.lora_idx if args.lora_idx is not None else \
            resolve_lora_idx(args.lora_label, args.lora_mapping)
        alphas = [float(a) for a in args.alphas.split(",")]
        synthesize_lora_sweep(
            args.pretrained, args.lora_vector, args.config, args.vocab, alphas,
            args.ref_audio, args.ref_text, transcripts, args.out_dir,
            args.nfe, args.seed, args.device, lora_idx=lora_idx,
        )
        return

    config_path = args.config or str(
        files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")
    )
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

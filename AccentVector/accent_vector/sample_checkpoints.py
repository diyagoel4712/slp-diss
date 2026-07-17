"""[RQ6 companion] Listen to the model as the accent forms: synthesise a FIXED
(ref clip, gen text) at every LoRA snapshot so you can scrub through step_<n>.wav
and hear the accent emerge (early vs late clips differ audibly).

Unlike the trainer's built-in log_samples (which uses whichever training batch
happens to be first, so clips aren't comparable across time), this holds the prompt
fixed. It runs post-hoc in the f5-tts inference env, decoupled from training.

The accent vector is the LoRA branch, so alpha-scaling is native: pass --lora-alpha
to fold tau at any strength (alpha=1 is the trained model; the RQ1 sweep is just
several --lora-alpha values). No merged checkpoints needed.

    python -m accent_vector.sample_checkpoints \
        --run-dir exps/F5TTS_v1_LoRA_british/<run> \
        --base-ckpt ckpts/F5TTS_v1_Base/model_1250000.pt \
        --ref-audio ref.wav --ref-text "..." \
        --gen-text "The quick brown fox jumps over the lazy dog." \
        --out-dir results/british/scrub
"""

import argparse
import json
import re
from pathlib import Path

import torch
import soundfile as sf
from omegaconf import OmegaConf
from hydra.utils import get_class

from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer
from f5_tts.infer.utils_infer import (
    load_vocoder,
    infer_process,
    preprocess_ref_audio_text,
)


def _snapshots(snap_dir):
    """Sorted [(step, path)] of lora_<step>.pt snapshots."""
    out = []
    for p in Path(snap_dir).glob("lora_*.pt"):
        m = re.search(r"(\d+)", p.stem)
        if m:
            out.append((int(m.group(1)), str(p)))
    return sorted(out)


def _base_state_dict(base_ckpt, device):
    """The base (non-LoRA) weights, keyed like the CFM's own state_dict."""
    ckpt = torch.load(base_ckpt, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    elif "ema_model_state_dict" in ckpt:
        sd = {k.replace("ema_model.", ""): v for k, v in ckpt["ema_model_state_dict"].items()
              if k not in ("initted", "step", "update")}
    else:
        sd = ckpt
    for key in ("mel_spec.mel_stft.mel_scale.fb", "mel_spec.mel_stft.spectrogram.window"):
        sd.pop(key, None)
    return sd


def run(run_dir, base_ckpt, ref_audio, ref_text, gen_text, out_dir,
        config_path=None, vocab_path=None, snap_dir=None, lora_alpha=None,
        lora_label=None, device=None):
    run_dir = Path(run_dir)
    config_path = config_path or (run_dir / "config.yaml")
    vocab_path = vocab_path or (run_dir / "vocab.txt")
    snap_dir = snap_dir or (run_dir / "ckpts" / "snapshots")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.load(str(config_path))
    arch = OmegaConf.to_container(cfg.model.arch, resolve=True)
    if lora_alpha is not None:  # native alpha-scaling of the accent vector
        arch["lora_alpha"] = float(lora_alpha)

    # lora_idx for this accent (default 0 = the single LoRA trained here).
    lora_idx = 0
    mapping = run_dir / "lora_mapping.json"
    if lora_label is not None and mapping.exists():
        lora_idx = json.load(open(mapping))[lora_label]

    snaps = _snapshots(snap_dir)
    if not snaps:
        raise SystemExit(f"no lora_<step>.pt snapshots in {snap_dir}")
    print(f"[scrub] {len(snaps)} snapshots, steps {snaps[0][0]}..{snaps[-1][0]}; "
          f"lora_alpha={arch.get('lora_alpha', 1.0)} lora_idx={lora_idx} device={device}")

    mel = cfg.model.mel_spec
    mel_spec_kwargs = dict(n_fft=mel.n_fft, hop_length=mel.hop_length, win_length=mel.win_length,
                           n_mel_channels=mel.n_mel_channels, target_sample_rate=mel.target_sample_rate,
                           mel_spec_type=mel.mel_spec_type)
    vocab_char_map, vocab_size = get_tokenizer(str(vocab_path), "custom")
    vocoder = load_vocoder(vocoder_name=mel.mel_spec_type, is_local=cfg.model.vocoder.is_local,
                           local_path=cfg.model.vocoder.local_path)

    model_cls = get_class(f"f5_tts.model.{cfg.model.backbone}")
    model = CFM(
        transformer=model_cls(**arch, text_num_embeds=vocab_size, mel_dim=mel.n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
        tokenized=cfg.model.get("tokenized", False),
        tokenizer=cfg.model.tokenizer,
    ).to(device)

    # Base fills the frozen backbone; LoRA keys stay at init (zero effect) until overlaid.
    base_sd = _base_state_dict(base_ckpt, device)
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    print(f"[scrub] base loaded (missing={len(missing)} lora keys, unexpected={len(unexpected)})")

    ref_audio_p, ref_text_p = preprocess_ref_audio_text(ref_audio, ref_text)

    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for step, path in snaps:
        lora_sd = torch.load(path, map_location=device, weights_only=True)
        lora_sd = lora_sd.get("lora_state_dict", lora_sd)
        model.load_state_dict(lora_sd, strict=False)  # overlay this step's accent vector
        model.eval()
        with torch.inference_mode():
            audio, sr, _ = infer_process(
                ref_audio_p, ref_text_p, gen_text, model, vocoder,
                mel_spec_type=mel.mel_spec_type, device=device, lora_idx=lora_idx,
            )
        wav = out_dir / f"step_{step}.wav"
        sf.write(str(wav), audio, sr)
        print(f"[scrub] step {step:>8} -> {wav}")
    print(f"[scrub] done; scrub through {out_dir}/step_*.wav")


def main():
    p = argparse.ArgumentParser(description="Synthesise a fixed prompt at every LoRA snapshot")
    p.add_argument("--run-dir", required=True, help="training run dir (has config.yaml, vocab.txt, ckpts/snapshots)")
    p.add_argument("--base-ckpt", required=True, help="base F5-TTS checkpoint (frozen backbone)")
    p.add_argument("--ref-audio", required=True)
    p.add_argument("--ref-text", required=True)
    p.add_argument("--gen-text", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--config", help="override path to config.yaml")
    p.add_argument("--vocab", help="override path to vocab.txt")
    p.add_argument("--snap-dir", help="override snapshots dir")
    p.add_argument("--lora-alpha", type=float, help="accent-vector strength (default: trained alpha)")
    p.add_argument("--lora-label", help="accent label to look up in lora_mapping.json (default idx 0)")
    p.add_argument("--device", help="cuda | cpu (default: auto)")
    a = p.parse_args()
    run(a.run_dir, a.base_ckpt, a.ref_audio, a.ref_text, a.gen_text, a.out_dir,
        config_path=a.config, vocab_path=a.vocab, snap_dir=a.snap_dir,
        lora_alpha=a.lora_alpha, lora_label=a.lora_label, device=a.device)


if __name__ == "__main__":
    main()

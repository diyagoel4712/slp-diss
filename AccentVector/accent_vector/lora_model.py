"""Shared machinery for the native-LoRA path (accent vector = LoRA branch).

The F5-TTS fork implements the accent vector as an *additive* LoRA branch whose
forward is ``out * lora_alpha`` (``f5_tts.model.modules.LoRALinear``), with
``lora_alpha`` a plain per-module attribute. That makes the paper's task-vector
arithmetic native, with **no checkpoint merge**:

    tau = theta_LoRA                          (Eq. 3: the trained LoRA weights)
    theta = theta_pre + alpha * tau           (Eq. 4: set lora_alpha = alpha)

So an accent-strength sweep builds the model ONCE (frozen base backbone + one
overlaid LoRA vector) and then rescales the branch in place before each synthesis:
``alpha = 0`` zeroes the branch (pure base = the sweep's baseline), ``alpha = 1``
is the trained strength. This is the counterpart, on the LoRA track, of
``extract_vector.compose`` on the full-fine-tune track.

Used by ``infer_accent`` (alpha sweep, fixed vector) and ``sample_checkpoints``
(fixed alpha, varying snapshot). Imports F5-TTS, so it is only loaded on the GPU
inference path -- the CPU geometry analyses never import it.
"""

import json
from pathlib import Path

import torch
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.model import CFM
from f5_tts.model.utils import get_tokenizer
from f5_tts.infer.utils_infer import load_vocoder


def base_state_dict(base_ckpt, device):
    """Base (non-LoRA) weights keyed like the CFM's own state_dict (EMA preferred).

    The frozen backbone; LoRA keys are absent here and stay at init (zero effect)
    until a vector is overlaid.
    """
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


def load_lora_state(lora_path, device="cpu"):
    """The trainable LoRA tensors from a snapshot/vector (unwraps ``lora_state_dict``)."""
    obj = torch.load(lora_path, map_location=device, weights_only=True)
    return obj.get("lora_state_dict", obj) if isinstance(obj, dict) else obj


def resolve_lora_idx(lora_label=None, mapping_path=None):
    """Accent label -> LoRA branch index via lora_mapping.json (default idx 0)."""
    if lora_label is not None and mapping_path and Path(mapping_path).exists():
        return json.load(open(mapping_path))[lora_label]
    return 0


def build_base_model(config_path, vocab_path, base_ckpt, device, lora_alpha=None):
    """Build the F5-TTS CFM from a LoRA config and load the frozen base backbone.

    ``lora_alpha`` (if given) overrides the config's value for every LoRA submodule
    at construction; leave it None to keep the config default and set it later with
    ``set_lora_alpha``. Returns ``(model, cfg, vocoder)``.
    """
    cfg = OmegaConf.load(str(config_path))
    arch = OmegaConf.to_container(cfg.model.arch, resolve=True)
    if lora_alpha is not None:
        arch["lora_alpha"] = float(lora_alpha)
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
    base_sd = base_state_dict(base_ckpt, device)
    missing, unexpected = model.load_state_dict(base_sd, strict=False)
    print(f"[lora] base loaded (missing={len(missing)} lora keys, unexpected={len(unexpected)})")
    return model, cfg, vocoder


def overlay_lora(model, lora_state):
    """Overlay a LoRA accent vector onto the (already base-loaded) model in place.

    Keys shared with a previous overlay are replaced, so this can be called
    repeatedly to swap in successive snapshots (the RQ6 trajectory).
    """
    model.load_state_dict(lora_state, strict=False)


def set_lora_alpha(model, alpha):
    """Scale the accent branch to strength ``alpha`` (Eq. 4) by setting lora_alpha on
    every LoRA submodule. Returns the number of submodules touched."""
    a = float(alpha)
    n = 0
    for m in model.modules():
        if hasattr(m, "lora_alpha"):
            m.lora_alpha = a
            n += 1
    return n

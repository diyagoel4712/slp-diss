"""Accent Vector extraction and layer masking (paper Section 3.2 / 3.3).

The Accent Vector is the parameter shift induced by fine-tuning F5-TTS on a
target accent/language. Fine-tuning here is **always LoRA** (the paper's recipe,
Eq. 3), so the vector IS the LoRA branch:

    tau_accent = theta_LoRA                                 (Eq. 1-3)
    theta      = theta_pre + alpha * tau_accent             (Eq. 4)

and the arithmetic is native -- ``lora_model.set_lora_alpha`` rescales the branch
in place, with no checkpoint merge. This module's job is to get tau OUT of a
training checkpoint when no ``lora_<step>.pt`` snapshot was written, plus the
flatten / layer-mask helpers the geometry analyses share.

Usage
-----
    # slice the LoRA vector out of a full training checkpoint
    python -m accent_vector.extract_vector extract-lora \
        --checkpoint exps/.../ckpts/model_last.pt \
        --out        vectors/dutch.pt
"""

import argparse
import os
from collections import OrderedDict

import torch


# --- flat <-> nested state-dict helpers (F5 checkpoints are nested dicts) ----
# We flatten so we can diff/scale every leaf tensor by a single string key, then
# unflatten back to the exact layout F5 expects. Separator matches the layout
# used by Expressive-Vectors so checkpoints round-trip byte-for-byte.
SEP = "@"


def flatten_state_dict(d, parent_key="", sep=SEP):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + str(k) if parent_key else str(k)
        if isinstance(v, (OrderedDict, dict)):
            items.extend(flatten_state_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def load_flat_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, (dict, OrderedDict)):
        raise ValueError(f"Unexpected checkpoint object in {path}: {type(checkpoint)}")
    return flatten_state_dict(checkpoint)


# --- LoRA recovery: pull tau straight out of a full checkpoint --------------
def _is_lora_key(key):
    """A flat state-dict key belongs to the LoRA branch iff some path component
    is a ``lora_*`` submodule (lora_q, lora_conv1, lora_linear, ...). LoRALinear
    stores only its low-rank ``encoders``/``decoders`` there -- no frozen base
    copy -- so these keys are exactly the trainable accent vector tau (Eq. 3)."""
    return any(part.startswith("lora_") for part in key.split("."))


def extract_lora(checkpoint_path, out_path, source="model", verbose=True):
    """Recover the LoRA accent vector from a FULL training checkpoint
    (model_last.pt / model_<step>.pt) when no ``lora_<step>.pt`` snapshot exists.

    The unmerged-LoRA fine-tune leaves the base weights frozen and puts tau in
    new ``lora_*`` keys, so the diff-based ``extract`` yields an empty vector.
    Here we instead slice those keys directly, reproducing byte-for-byte what
    ``Trainer.save_lora_snapshot`` would have written -- a dict consumable by
    ``lora_model.load_lora_state`` / ``infer_accent``.

    ``source`` picks which weights to read: ``"model"`` (the raw trained model,
    matching what save_lora_snapshot captured) or ``"ema"`` (the EMA-smoothed
    copy, often preferred for a final deliverable vector).
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"unexpected checkpoint layout in {checkpoint_path}: not a dict")

    if source == "ema":
        sd = ckpt.get("ema_model_state_dict")
        if sd is None:
            raise RuntimeError(f"no ema_model_state_dict in {checkpoint_path}")
        # ema_pytorch prefixes the tracked weights with 'ema_model.'; drop its
        # bookkeeping (initted/step) and the shadowed 'online_model.' copy.
        sd = {k[len("ema_model."):]: v for k, v in sd.items() if k.startswith("ema_model.")}
    elif source == "model":
        sd = ckpt.get("model_state_dict")
        if sd is None:
            raise RuntimeError(f"no model_state_dict in {checkpoint_path}")
    else:
        raise ValueError(f"source must be 'model' or 'ema', got {source!r}")

    lora_state = {k: v.detach().cpu() for k, v in sd.items()
                  if isinstance(v, torch.Tensor) and _is_lora_key(k)}
    if not lora_state:
        raise RuntimeError(
            f"no lora_* tensors found in {source} weights of {checkpoint_path}; "
            "is this an unmerged-LoRA checkpoint?"
        )

    update = ckpt.get("update")
    if verbose:
        n_params = sum(t.numel() for t in lora_state.values())
        print(f"[extract-lora] {len(lora_state)} LoRA tensors ({n_params:,} params) "
              f"from {source} weights, update={update}")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save({"lora_state_dict": lora_state, "update": update}, out_path)
    print(f"[extract-lora] saved LoRA accent vector -> {out_path}")
    return out_path


# --- layer masking (shared with lora_model.set_lora_alpha) ------------------
def _key_selected(key, include, exclude):
    """Layer masking: keep ``key`` iff it matches an include substring (when the
    include list is non-empty) and no exclude substring. Shared with
    ``lora_model.set_lora_alpha``, which masks LoRA submodules by the same rule
    -- the primitive behind layer localisation and layer-targeted scaling."""
    if include and not any(s in key for s in include):
        return False
    if exclude and any(s in key for s in exclude):
        return False
    return True


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Accent Vector extraction (LoRA). Fine-tuning is always LoRA, so the "
                    "vector is the LoRA branch; there is no full-weight diff or merge.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_el = sub.add_parser(
        "extract-lora",
        help="recover the LoRA vector (lora_state_dict) from a full checkpoint when no snapshot exists",
    )
    p_el.add_argument("--checkpoint", required=True,
                      help="full training checkpoint (e.g. model_last.pt) holding the lora_* keys")
    p_el.add_argument("--out", required=True, help="output path for the LoRA accent vector")
    p_el.add_argument("--source", choices=("model", "ema"), default="model",
                      help="which weights to slice: 'model' (raw, = save_lora_snapshot) or 'ema'")
    return parser


def main():
    args = _build_parser().parse_args()
    if args.command == "extract-lora":
        extract_lora(args.checkpoint, args.out, source=args.source)


if __name__ == "__main__":
    main()

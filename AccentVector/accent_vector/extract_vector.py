"""Accent Vector extraction and arithmetic (paper Section 3.2 / 3.3).

The Accent Vector is the parameter shift induced by fine-tuning F5-TTS on a
target accent/language:

    tau_accent = theta_ft - theta_pre                       (Eq. 1)

When the fine-tune is a LoRA, this difference is exactly the LoRA delta
(Eq. 3); when it is a full fine-tune, it is the full-weight delta. Either way
the arithmetic downstream is identical, so we extract the vector as a plain
checkpoint difference and stay agnostic to how theta_ft was produced.

Arithmetic:

    theta = theta_pre + alpha * tau_accent                  (Eq. 4, scaling)
    theta = theta_pre + sum_i alpha_i * tau_accent(i)       (Eq. 5-6, mixing)

The resulting checkpoint is written in the same nested layout F5-TTS produces,
so it loads directly with ``f5_tts.infer.utils_infer.load_model`` / EMA
selection, exactly like a normal fine-tuned checkpoint.

Usage
-----
    # Eq. 1-3: extract tau from a fine-tuned checkpoint
    python -m accent_vector.extract_vector extract \
        --pretrained ckpts/F5TTS_v1_Base/model_1250000.pt \
        --finetuned  ckpts/british/model_60000.pt \
        --out        vectors/british.pt

    # Eq. 4: scale a single accent vector (alpha sweep does this per alpha)
    python -m accent_vector.extract_vector compose \
        --pretrained ckpts/F5TTS_v1_Base/model_1250000.pt \
        --vector vectors/british.pt --alpha 0.6 \
        --out    ckpts/british/accent_a0.6.pt

    # Eq. 5-6: linearly compose two accents (mixed-accent speech)
    python -m accent_vector.extract_vector compose \
        --pretrained ckpts/F5TTS_v1_Base/model_1250000.pt \
        --vector vectors/spanish.pt --alpha 0.5 \
        --vector vectors/british.pt --alpha 0.5 \
        --out    ckpts/mixed/spanish+british.pt
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


def unflatten_state_dict(flat_dict, sep=SEP):
    unflat = dict()
    for flat_key, value in flat_dict.items():
        parts = flat_key.split(sep)
        d = unflat
        for part in parts[:-1]:
            if part not in d or not isinstance(d[part], dict):
                d[part] = dict()
            d = d[part]
        d[parts[-1]] = value
    return unflat


def load_flat_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, (dict, OrderedDict)):
        raise ValueError(f"Unexpected checkpoint object in {path}: {type(checkpoint)}")
    return flatten_state_dict(checkpoint)


# --- Eq. 1-3: extract the accent vector -------------------------------------
def compute_task_vector(pretrained_flat, finetuned_flat, verbose=True):
    """tau = theta_ft - theta_pre over every matching float tensor.

    Non-float leaves (step counters, optimizer bookkeeping), missing keys, and
    shape mismatches are skipped: they carry no accent direction and are simply
    inherited from theta_pre when a model is rebuilt.
    """
    diff = {}
    norms = {}
    for key, p_pre in pretrained_flat.items():
        p_ft = finetuned_flat.get(key)
        if p_ft is None:
            if verbose:
                print(f"[extract] skip (absent in finetuned): {key}")
            continue
        if not isinstance(p_pre, torch.Tensor) or not isinstance(p_ft, torch.Tensor):
            continue
        if p_pre.shape != p_ft.shape:
            if verbose:
                print(f"[extract] skip (shape {p_pre.shape} vs {p_ft.shape}): {key}")
            continue
        if not torch.is_floating_point(p_pre):
            continue
        d = p_ft.float() - p_pre.float()
        diff[key] = d
        norms[key] = d.pow(2).mean().sqrt().item()  # RMS magnitude of the shift
    if verbose:
        moved = sorted(norms.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print(f"[extract] captured {len(diff)} tensors; top-shifted layers:")
        for k, n in moved:
            print(f"           ||delta||_rms = {n:.6f}  {k}")
    return diff, norms


def extract(pretrained_path, finetuned_path, out_path, verbose=True):
    pre = load_flat_checkpoint(pretrained_path)
    ft = load_flat_checkpoint(finetuned_path)
    diff, _ = compute_task_vector(pre, ft, verbose=verbose)
    if not diff:
        raise RuntimeError(
            "Empty task vector: no matching float tensors moved between the two "
            "checkpoints. If you fine-tuned with unmerged LoRA adapters, the "
            "delta lives in new keys absent from the base model -- merge the "
            "adapters into the base weights before extracting."
        )
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(unflatten_state_dict(diff), out_path)
    print(f"[extract] saved accent vector ({len(diff)} tensors) -> {out_path}")
    return out_path


# --- Eq. 4-6: build an accent-modified model --------------------------------
def _key_selected(key, include, exclude):
    """Layer masking: keep ``key`` iff it matches an include substring (when the
    include list is non-empty) and no exclude substring. Enables scaling only a
    subset of layers -- the primitive behind layer localisation (RQ3.4) and
    layer-targeted scaling (RQ4)."""
    if include and not any(s in key for s in include):
        return False
    if exclude and any(s in key for s in exclude):
        return False
    return True


def compose(pretrained_path, vectors, out_path, include=None, exclude=None, verbose=True):
    """theta_pre + sum_i alpha_i * tau_i, optionally over a subset of layers.

    ``vectors`` is a list of ``(vector_path, alpha)`` or
    ``(vector_path, alpha, per_vector_include)``. A single pair with alpha in
    [0, 1] realises the strength sweep (Eq. 4); several pairs realise
    mixed-accent composition (Eq. 5-6). alpha == 0 for a single unmasked vector
    reproduces the pretrained model exactly (the sweep's baseline).

    ``include`` / ``exclude`` are lists of substrings matched against flattened
    parameter keys; a per-vector include (3rd tuple element) overrides the
    global ``include`` for that vector. Keys not selected keep their pretrained
    value, so a mask scales only the chosen layers and leaves the rest at
    theta_pre.
    """
    result = load_flat_checkpoint(pretrained_path)  # start from theta_pre
    for spec in vectors:
        vector_path, alpha = spec[0], spec[1]
        inc = spec[2] if len(spec) > 2 and spec[2] is not None else include
        tau = load_flat_checkpoint(vector_path)
        applied = skipped_mask = 0
        for key, delta in tau.items():
            if key not in result or not isinstance(result[key], torch.Tensor):
                if verbose:
                    print(f"[compose] vector key not in base, skipping: {key}")
                continue
            if not _key_selected(key, inc, exclude):
                skipped_mask += 1
                continue
            result[key] = result[key].float() + alpha * delta.float()
            applied += 1
        mask_note = f", {skipped_mask} masked out" if (inc or exclude) else ""
        print(f"[compose] applied {vector_path} alpha={alpha} ({applied} tensors{mask_note})")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    torch.save(unflatten_state_dict(result), out_path)
    print(f"[compose] saved accent-modified model -> {out_path}")
    return out_path


def _build_parser():
    parser = argparse.ArgumentParser(description="Accent Vector extraction and arithmetic")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ex = sub.add_parser("extract", help="tau = theta_ft - theta_pre (Eq. 1-3)")
    p_ex.add_argument("--pretrained", required=True, help="base F5 checkpoint (theta_pre)")
    p_ex.add_argument("--finetuned", required=True, help="fine-tuned checkpoint (theta_ft)")
    p_ex.add_argument("--out", required=True, help="output path for the accent vector")

    p_co = sub.add_parser("compose", help="theta_pre + sum a_i * tau_i (Eq. 4-6)")
    p_co.add_argument("--pretrained", required=True, help="base F5 checkpoint (theta_pre)")
    p_co.add_argument(
        "--vector", action="append", default=[], required=True,
        help="path to an accent vector; repeat for mixed accents",
    )
    p_co.add_argument(
        "--alpha", action="append", default=[], type=float, required=True,
        help="strength coefficient for the corresponding --vector; repeat to match",
    )
    p_co.add_argument("--out", required=True, help="output path for the accent-modified model")
    p_co.add_argument("--include", action="append", default=[],
                      help="only scale layers whose key contains this substring (repeatable)")
    p_co.add_argument("--exclude", action="append", default=[],
                      help="never scale layers whose key contains this substring (repeatable)")
    return parser


def main():
    args = _build_parser().parse_args()
    if args.command == "extract":
        extract(args.pretrained, args.finetuned, args.out)
    elif args.command == "compose":
        if len(args.vector) != len(args.alpha):
            raise SystemExit(
                f"got {len(args.vector)} --vector but {len(args.alpha)} --alpha; "
                "they must be paired one-to-one"
            )
        compose(args.pretrained, list(zip(args.vector, args.alpha)), args.out,
                include=args.include or None, exclude=args.exclude or None)


if __name__ == "__main__":
    main()

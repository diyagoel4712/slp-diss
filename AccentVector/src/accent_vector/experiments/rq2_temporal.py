"""[RQ2 / Tier 1] Temporal trajectory of the accent vector during fine-tuning.

For a sequence of fine-tuning checkpoints theta_t, form tau_t = theta_t - theta_pre
and track -- at near-zero extra cost, pure CPU vector math over checkpoints you
already save:

  magnitude      ||tau_t||             how fast the delta grows
  direction      cos(tau_t, tau_final) whether the accent DIRECTION stabilises
                                        early. If it does, the direction is
                                        learnable from little optimisation and
                                        alpha supplies the remaining intensity.
  convergence    interpolated step at which cos(tau_t, tau_final) first crosses a
                 threshold (default 0.95) -- one comparable "the direction is set
                 by step N" number (reuses shared.leakage_onset).

SCOPE: this is the OPTIMISATION trajectory, NOT data efficiency -- F5 fine-tunes
many epochs over the same corpus, so step != amount of data. The "how much DATA"
question needs separate LoRAs trained on data fractions (Tier 2), which is out of
scope here. For the downstream-EFFECT trajectory, extract a few checkpoints'
vectors (extract_vector) and run the alpha-sweep + rq1/rq3 on them -- no new code.

Tip: pass ``--include ema_model_state_dict`` to track the weights inference
actually uses; optimizer moments are dropped by default (they aren't part of the
accent vector and would pollute magnitude/direction).

The accent vector IS the LoRA weights, so tau_t is read straight from the
``lora_<step>.pt`` snapshots -- point ``--ckpt-dir`` at the snapshots dir:

    python -m accent_vector.experiments.rq2_temporal \
        --ckpt-dir exps/F5TTS_v1_LoRA_dutch/<run>/ckpts/snapshots \
        --out-csv results/per-accent/dutch/temporal.csv
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from accent_vector import shared
from accent_vector.extract_vector import _key_selected

DEFAULT_EXCLUDE = ["optimizer"]  # drop Adam moments; not part of the accent vector


def _step(name):
    """Training step from a checkpoint filename (model_60000.pt -> 60000)."""
    m = re.search(r"(\d+)", Path(name).stem)
    return int(m.group(1)) if m else None


def collect_checkpoints(ckpt_dir, prefix=None):
    """Sorted [(step, path)] for <prefix><step>.pt (default any model_<step>.pt),
    skipping derived files."""
    out = []
    for p in Path(ckpt_dir).glob("*.pt"):
        if prefix is not None and not p.name.startswith(prefix):
            continue
        s = _step(p.name)
        if s is not None and not any(x in p.name for x in ("interpolated", "diff", "accent_a")):
            out.append((s, str(p)))
    return sorted(out)


def _lora_flat(ckpt_path):
    """Flat dict of a LoRA snapshot's float tensors. The accent vector IS the LoRA
    weights (tau = theta_LoRA, paper Eq. 3); zero-init decoders make the pretrained
    baseline implicitly the zero vector, so no theta_pre diff is needed."""
    import torch

    obj = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = obj.get("lora_state_dict", obj) if isinstance(obj, dict) else obj
    return {k: v for k, v in sd.items()
            if hasattr(v, "is_floating_point") and v.is_floating_point()}


def _vector_1d_lora(ckpt_path, include, exclude):
    """tau_t straight from the LoRA snapshot weights (no theta_pre diff)."""
    flat = _lora_flat(ckpt_path)
    keys = sorted(k for k in flat if _key_selected(k, include, exclude))
    if not keys:
        raise SystemExit(f"no selected float LoRA tensors in {ckpt_path} "
                         f"(include={include}, exclude={exclude})")
    vec = np.concatenate([flat[k].detach().cpu().float().numpy().ravel() for k in keys])
    return keys, vec


def run(ckpt_dir, out_csv, threshold=0.95, include=None, exclude=None,
        final_ckpt=None):
    exclude = exclude if exclude is not None else DEFAULT_EXCLUDE
    ckpts = collect_checkpoints(ckpt_dir, prefix="lora_")
    if not ckpts:
        raise SystemExit(f"no lora_<step>.pt snapshots found in {ckpt_dir}")
    vec_of = lambda path: _vector_1d_lora(path, include, exclude)

    final_path = final_ckpt or ckpts[-1][1]
    keys_final, v_final = vec_of(final_path)
    n_final = float(np.linalg.norm(v_final))
    print(f"[temporal] final = {final_path}  ||tau_final||={n_final:.4f}  dim={v_final.size}")

    rows = []
    for step, path in ckpts:
        keys, v = vec_of(path)
        if keys != keys_final:
            raise SystemExit(f"parameter-key mismatch at {path}; checkpoints must "
                             f"share architecture with the final checkpoint")
        mag = float(np.linalg.norm(v))
        cos = float(v @ v_final / (mag * n_final)) if mag > 0 and n_final > 0 else float("nan")
        rows.append({"step": step, "magnitude": mag,
                     "mag_frac_final": mag / n_final if n_final else float("nan"),
                     "cos_to_final": cos})
        print(f"[temporal] step={step:>8}  ||tau||={mag:.4f} "
              f"({mag / n_final:.2f} of final)  cos_to_final={cos:.4f}")

    steps = [r["step"] for r in rows]
    dir_step = shared.leakage_onset(steps, [r["cos_to_final"] for r in rows], threshold, rising=True)
    mag_step = shared.leakage_onset(steps, [r["mag_frac_final"] for r in rows], threshold, rising=True)
    print(f"[temporal] direction reaches cos>={threshold} at step ~{dir_step}; "
          f"magnitude reaches {threshold} of final at step ~{mag_step}  "
          f"(direction settling before magnitude => alpha can substitute for training)")

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", "magnitude", "mag_frac_final", "cos_to_final"])
        w.writeheader()
        w.writerows(rows)
        f.write(f"# direction_converge_step(cos>={threshold})={dir_step}, "
                f"magnitude_converge_step={mag_step}\n")
    print(f"[temporal] wrote {out_csv}")


def main():
    p = argparse.ArgumentParser(description="RQ2/Tier1 accent-vector fine-tuning trajectory")
    p.add_argument("--ckpt-dir", required=True,
                   help="dir of lora_<step>.pt snapshots")
    p.add_argument("--final-ckpt", help="reference tau_final (default: highest-step checkpoint)")
    p.add_argument("--threshold", type=float, default=0.95, help="convergence threshold")
    p.add_argument("--include", action="append", default=[],
                   help="only these key substrings (e.g. ema_model_state_dict)")
    p.add_argument("--exclude", action="append", default=None,
                   help="drop these key substrings (default: optimizer)")
    p.add_argument("--out-csv", required=True)
    a = p.parse_args()
    run(a.ckpt_dir, a.out_csv, threshold=a.threshold,
        include=a.include or None, exclude=a.exclude, final_ckpt=a.final_ckpt)


if __name__ == "__main__":
    main()

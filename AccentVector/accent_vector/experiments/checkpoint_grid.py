"""[RQ1 x RQ6 -- behavioural trajectory, GPU] Run the alpha sweep at SEVERAL
training checkpoints of one accent, so the downstream metrics can be compared at
matched alpha across training time (25k vs 45k vs ...).

This is the synthesis half of the "does the accent arrive before the language?"
question. ``rq6_temporal`` tracks the vector in WEIGHT space (||tau_t||, direction);
this produces the OUTPUT the model actually generates at each checkpoint, which
``rq1_reproduction`` (per checkpoint) then scores and ``rq6_behavioural`` compares.

For each requested step it resolves that step's LoRA accent vector and runs the
native LoRA alpha sweep (reusing ``infer_accent.synthesize_lora_sweep``) into

    <out-root>/step_<step>/alpha_<a>/utt####.wav

-- the same alpha_<a>/ layout as a single sweep, one level deeper, so every rq*
module reads a ``step_<step>/`` dir exactly like a normal sweep dir.

Vector resolution per step (first that exists wins):
  1. ``<snap-dir>/lora_<step>.pt``          -- the cheap LoRA snapshot (preferred)
  2. ``<ckpt-dir>/model_<step>.pt``         -- full checkpoint; the lora_* keys are
                                               sliced out on the fly (extract_lora)
  3. explicit ``--vector step=path``        -- overrides 1-2 for that step

Hold everything else fixed across steps (reference, transcripts, alpha grid, seed,
lora_idx) so training step is the only thing varying -- ideally the NEUTRAL
native-English reference (REF_KIND=native) so an accent rise is unambiguously the
vector, not cloning (see [[inference-reference-native-l1]]).

    python -m accent_vector.experiments.checkpoint_grid \
        --pretrained ckpts/F5TTS_v1_Base/model_1250000.pt \
        --config exps/.../config.yaml --vocab exps/.../vocab.txt \
        --snap-dir exps/F5TTS_v1_LoRA_dutch/<run>/ckpts/snapshots \
        --steps 5000,15000,25000,45000 \
        --ref-audio refs/native_ga.wav --ref-text-file refs/native_ga.txt \
        --transcripts transcripts/eval_transcripts.txt \
        --alphas 0,0.25,0.5,0.75,1.0 \
        --out-root results/dutch/native/by_step
"""

import argparse
import os
import tempfile
from pathlib import Path


def resolve_vectors(steps, snap_dir, ckpt_dir, explicit, tmp):
    """[(step, lora_vector_path)] for each requested step, extracting the LoRA
    vector from a full model_<step>.pt into ``tmp`` when only that exists.
    ``explicit`` is a dict {step: path} overriding discovery for that step."""
    from accent_vector.extract_vector import extract_lora

    out = []
    for step in steps:
        if step in explicit:
            out.append((step, explicit[step]))
            continue
        snap = Path(snap_dir) / f"lora_{step}.pt" if snap_dir else None
        if snap and snap.exists():
            out.append((step, str(snap)))
            continue
        full = Path(ckpt_dir) / f"model_{step}.pt" if ckpt_dir else None
        if full and full.exists():
            dest = os.path.join(tmp, f"lora_{step}.pt")
            extract_lora(str(full), dest, source="model", verbose=False)
            out.append((step, dest))
            continue
        raise SystemExit(
            f"no vector for step {step}: looked for {snap} and {full}. Pass "
            f"--vector {step}=<path>, or point --snap-dir/--ckpt-dir at the run."
        )
    return out


def run(pretrained, config, vocab, vectors, ref_audio, ref_text, transcripts,
        alphas, out_root, nfe, seed, device, lora_idx, include, exclude):
    from accent_vector.infer_accent import load_transcripts, synthesize_lora_sweep

    tx = load_transcripts(transcripts)
    for step, vec in vectors:
        out_dir = os.path.join(out_root, f"step_{step}")
        print(f"[ckpt-grid] step={step} vector={vec} -> {out_dir}")
        synthesize_lora_sweep(
            pretrained, vec, config, vocab, alphas,
            ref_audio, ref_text, tx, out_dir,
            nfe, seed, device, lora_idx=lora_idx, include=include, exclude=exclude,
        )
    print(f"[ckpt-grid] done; {len(vectors)} checkpoints under {out_root}/step_*/")


def main():
    p = argparse.ArgumentParser(description="Alpha sweep across training checkpoints (RQ1 x RQ6)")
    p.add_argument("--pretrained", required=True, help="base checkpoint (theta_pre)")
    p.add_argument("--config", required=True, help="training run config.yaml")
    p.add_argument("--vocab", required=True, help="training run vocab.txt")
    p.add_argument("--steps", help="comma-separated training steps, e.g. 5000,15000,25000,45000")
    p.add_argument("--snap-dir", help="dir of lora_<step>.pt snapshots")
    p.add_argument("--ckpt-dir", help="dir of full model_<step>.pt checkpoints (extract-lora fallback)")
    p.add_argument("--vector", action="append", default=[],
                   help="step=path override for a specific checkpoint (repeatable)")
    p.add_argument("--ref-audio", required=True, help="reference clip, fixed across all steps")
    p.add_argument("--ref-text", help="reference transcript (literal)")
    p.add_argument("--ref-text-file", help="reference transcript read from this file")
    p.add_argument("--transcripts", required=True, help="English transcripts, one per line")
    p.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0", help="comma-separated strengths")
    p.add_argument("--out-root", required=True, help="results root; each step -> <root>/step_<step>/")
    p.add_argument("--nfe", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--lora-idx", type=int, default=0)
    p.add_argument("--include-layers", action="append", default=[],
                   help="only scale layers whose name contains this substring (RQ4)")
    p.add_argument("--exclude-layers", action="append", default=[],
                   help="never scale layers whose name contains this substring (RQ4)")
    a = p.parse_args()

    explicit = {}
    for pair in a.vector:
        k, v = pair.split("=", 1)
        explicit[int(k)] = v
    steps = [int(s) for s in a.steps.split(",")] if a.steps else []
    steps = sorted(set(steps) | set(explicit))
    if not steps:
        raise SystemExit("give --steps and/or --vector step=path")

    if a.ref_text_file:
        ref_text = Path(a.ref_text_file).read_text().strip()
    elif a.ref_text is not None:
        ref_text = a.ref_text
    else:
        raise SystemExit("give --ref-text or --ref-text-file")

    alphas = [float(x) for x in a.alphas.split(",")]

    with tempfile.TemporaryDirectory() as tmp:
        vectors = resolve_vectors(steps, a.snap_dir, a.ckpt_dir, explicit, tmp)
        run(a.pretrained, a.config, a.vocab, vectors, a.ref_audio, ref_text,
            a.transcripts, alphas, a.out_root, a.nfe, a.seed, a.device,
            a.lora_idx, a.include_layers or None, a.exclude_layers or None)


if __name__ == "__main__":
    main()

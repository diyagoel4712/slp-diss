"""Accent-controlled inference with F5-TTS (paper Section 3.4 + Figure 3).

For each strength coefficient alpha, we build theta_pre + alpha * tau_accent and
synthesize a fixed set of English transcripts (input text to be synthesised). The
reference clip is held fixed *within* a sweep, so the accent vector is the only thing
that varies as alpha climbs. The sweep runs between two exact anchors:
**alpha=0 = theta_pre** (the pretrained model cloning the accent from the reference
alone, no fine-tuning) and **alpha=1 = theta_ft** (the fully fine-tuned model, full
accent-vector impact).

The reference *kind* is itself a deliberately-varied experimental condition compared
*across* sweeps (see scripts/eddie_infer_sweep.sh REF_KIND). It is NOT a fixed invariant:

* **L1 reference** (target accent's native-language clip; paper-faithful cloning, see
  ADAPTATION_PLAN.md gotcha #3). Because F5-TTS has no language-ID token or
  speaker/perceiver factorisation, it clones whatever accent is in this clip, so accent
  at alpha=0 comes from BOTH the reference and (as alpha rises) the vector -- the
  confounded baseline.
* **Neutral native-English reference** (a clip whose accent is NOT the target). At
  alpha=0 the output is neutral English, so any target-accent signal emerging as alpha
  climbs is attributable to the VECTOR alone -- the decoupling control that isolates the
  fine-tuning's contribution from reference cloning.

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
import torch
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
                   transcripts, out_dir, nfe_step, seed, device, lora_idx=None,
                   shard_index=0, shard_count=1):
    """Synthesize each transcript to ``<out_dir>/utt####.wav``.

    ``shard_count`` > 1 splits the work across parallel jobs: this call renders
    only transcripts whose GLOBAL index ``idx % shard_count == shard_index`` but
    keeps the global ``utt{idx}`` name -- so shards write DISJOINT files into the
    same dir (safe under exist_ok), reassemble into one complete alpha_<a>/, and
    the utt#### <-> transcript-index mapping the eval reads is unchanged. Per-utt
    RNG is reseeded to ``seed`` regardless of shard, so a clip is byte-identical
    whether or not it was sharded."""
    os.makedirs(out_dir, exist_ok=True)
    ref_audio, ref_text = preprocess_ref_audio_text(ref_audio, ref_text)
    if lora_idx is not None and not torch.is_tensor(lora_idx):
        # dit.py's forward always does lora_idx[0]; resolve_lora_idx/--lora-idx
        # give a plain int, which trained via collate_fn's tensor would never be.
        lora_idx = torch.tensor([lora_idx], device=device)
    n = 0
    for idx, gen_text in enumerate(transcripts):
        if shard_count > 1 and idx % shard_count != shard_index:
            continue
        # this fork's infer_process has no seed kwarg (unlike stock F5-TTS);
        # seed identically via the RNG directly before each call instead.
        torch.manual_seed(seed)
        wave, sr, _ = infer_process(
            ref_audio, ref_text, gen_text, model, vocoder,
            mel_spec_type=mel_spec_type, nfe_step=nfe_step, device=device,
            lora_idx=lora_idx,
        )
        sf.write(os.path.join(out_dir, f"utt{idx:04d}.wav"), wave, sr)
        n += 1
    shard_note = f" (shard {shard_index}/{shard_count})" if shard_count > 1 else ""
    print(f"[infer] wrote {n} clips{shard_note} -> {out_dir}")


def synthesize_lora_sweep(pretrained, lora_vector, config_path, vocab, alphas,
                          ref_audio, ref_text, transcripts, out_dir,
                          nfe, seed, device, lora_idx=0, include=None, exclude=None,
                          shard_index=0, shard_count=1):
    """Native LoRA alpha sweep: build the base+LoRA model once, then rescale the
    accent branch to each alpha in place (no per-alpha checkpoint merge). Writes
    ``<out_dir>/alpha_<a>/utt####.wav`` -- the same layout the merged path uses,
    so every downstream analysis reads it identically.

    ``include`` / ``exclude`` (substrings over LoRA submodule names) restrict the
    scaling to a subset of layers -- layer-targeted accent transfer (RQ4); masked
    submodules stay at theta_pre. See ``lora_model.set_lora_alpha``."""
    from accent_vector.lora_model import (
        build_base_model, load_lora_state, overlay_lora, set_lora_alpha,
    )

    model, cfg, vocoder = build_base_model(config_path, vocab, pretrained, device)
    mel_spec_type = cfg.model.mel_spec.mel_spec_type
    overlay_lora(model, load_lora_state(lora_vector, device))
    model.eval()
    for alpha in alphas:
        n_scaled, n_masked = set_lora_alpha(model, alpha, include=include, exclude=exclude)
        mask_note = f", {n_masked} masked->theta_pre" if n_masked else ""
        print(f"[infer:lora] alpha={alpha} on {n_scaled} LoRA submodules{mask_note} "
              f"(lora_idx={lora_idx})")
        synthesize_set(
            model, vocoder, mel_spec_type, ref_audio, ref_text,
            transcripts, os.path.join(out_dir, f"alpha_{alpha}"),
            nfe, seed, device, lora_idx=lora_idx,
            shard_index=shard_index, shard_count=shard_count,
        )


def main():
    parser = argparse.ArgumentParser(description="Accent-controlled F5-TTS inference")
    parser.add_argument("--config", default=None,
                        help="F5 model config yaml (default: packaged F5TTS_v1_Base.yaml)")
    parser.add_argument("--vocab", default="", help="vocab.txt (default: packaged pretrained vocab)")
    parser.add_argument("--ref-audio", required=True,
                        help="reference clip to clone, fixed within the sweep (L1 or a "
                             "neutral native-English control -- see module docstring)")
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

    # layer masking (RQ4 / RQ3.4): scale only a subset of the vector's layers.
    parser.add_argument("--include-layers", action="append", default=[],
                        help="only scale layers whose name/key contains this substring "
                             "(repeatable; e.g. attn, ff, conv). Others stay at theta_pre.")
    parser.add_argument("--exclude-layers", action="append", default=[],
                        help="never scale layers whose name/key contains this substring "
                             "(repeatable; e.g. text_embed to drop the content/language path).")

    # transcript sharding for multi-GPU fan-out: split the transcript list across
    # <shard-count> parallel jobs; this one renders indices == shard-index (mod count).
    parser.add_argument("--shard-index", type=int, default=0, help="this shard's id, 0..count-1")
    parser.add_argument("--shard-count", type=int, default=1, help="total shards (1 = no sharding)")

    args = parser.parse_args()
    include = args.include_layers or None
    exclude = args.exclude_layers or None
    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit(f"--shard-index must be in [0, {args.shard_count}); got {args.shard_index}")

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
            include=include, exclude=exclude,
            shard_index=args.shard_index, shard_count=args.shard_count,
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
            shard_index=args.shard_index, shard_count=args.shard_count,
        )
        return

    if not (args.pretrained and args.vector and args.alphas):
        raise SystemExit("sweep mode needs --pretrained, --vector and --alphas (or use --ckpt)")

    alphas = [float(a) for a in args.alphas.split(",")]
    with tempfile.TemporaryDirectory() as tmp:
        for alpha in alphas:
            ckpt = os.path.join(tmp, f"accent_a{alpha}.pt")
            compose(args.pretrained, [(args.vector, alpha)], ckpt,
                    include=include, exclude=exclude, verbose=False)
            model, mel_spec_type = build_model(config_path, ckpt, args.vocab, args.device)
            synthesize_set(
                model, vocoder, mel_spec_type, args.ref_audio, args.ref_text,
                transcripts, os.path.join(args.out_dir, f"alpha_{alpha}"),
                args.nfe, args.seed, args.device,
                shard_index=args.shard_index, shard_count=args.shard_count,
            )
            os.remove(ckpt)


if __name__ == "__main__":
    main()

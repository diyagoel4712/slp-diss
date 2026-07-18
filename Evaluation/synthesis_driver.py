"""Drive each TTS model over the L2-ARCTIC evaluation grid.

Writes one wav per (accent, speaker, utterance) cell to the exact layout that
`run_eval.py` scores:

    SOTA_models_experiments/<model>/outputs/<model>/<accent>/<speaker>/<utt_id>.wav

The grid + per-speaker reference logic live in `eval_config.py` (shared with `run_eval.py`).
Run this ONCE PER MODEL, each inside that model's own environment, because the
models have conflicting dependencies (same reason the metrics are split across envs):

    # in the coqui/XTTS env:
    python synthesis_driver.py --model xtts
    # in the f5-tts env:
    python synthesis_driver.py --model f5tts
    # in the CosyVoice3 env:
    python synthesis_driver.py --model cosyvoice3
    # baselines:
    python synthesis_driver.py --model vits      # VCTK speaker-id, native-accent baseline
    python synthesis_driver.py --model parler    # NL accent-description baseline

Three model families (see eval_config.MODEL_FAMILY):
  * clone       (xtts/f5tts/cosyvoice3): accent comes from a per-speaker held-out
                L2-ARCTIC reference clip (eval_config.REFERENCE_UTT, NOT one of the 10).
  * speaker_id  (vits): can only emit a trained VCTK speaker -> one fixed `sid`, same
                audio for every L2 speaker (deduplicated, then copied across cells).
  * description (parler): accent given as text; audio depends only on (accent, gender,
                utterance), so it is deduplicated too.

Useful flags:
  --ref-utt arctic_a0003   override the enrollment clip for the cloning models
  --limit 4                only the first N cells (quick smoke test)
  --overwrite              regenerate clips that already exist
  --dry-run                list what would be written; imports no model libraries
"""
import argparse
import contextlib
import os
import sys
from pathlib import Path

import eval_config as cfg


@contextlib.contextmanager
def _chdir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _mkout(it):
    it["out"].parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------------------
# cloning models: per-speaker held-out reference clip supplies voice + accent
# --------------------------------------------------------------------------------------
def _run_clone(items, infer_one):
    prompts = cfg.load_prompts()
    enroll = {}
    for it in items:
        spk = it["speaker"]
        if spk not in enroll:
            enroll[spk] = cfg.enrollment(spk, prompts)   # extracts ref wav on demand
        ref_wav, ref_text = enroll[spk]
        _mkout(it)
        infer_one(it, str(ref_wav), ref_text)
        print("saved", it["out"])


def synth_xtts(items):
    # Use the TTS package installed in coqui/.venv -- NOT the local coqui/TTS source tree:
    # that source pins an old transformers (imports BeamSearchScorer) and breaks against the
    # env's newer transformers. The installed package is what coqui/synthesise.py uses.
    from TTS.api import TTS
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

    def infer_one(it, ref_wav, ref_text):
        tts.tts_to_file(text=it["text"], speaker_wav=ref_wav,
                        language="en", file_path=str(it["out"]))

    _run_clone(items, infer_one)


def synth_f5tts(items):
    from f5_tts.api import F5TTS
    model = F5TTS(model="F5TTS_v1_Base")

    def infer_one(it, ref_wav, ref_text):
        model.infer(ref_file=ref_wav, ref_text=ref_text, gen_text=it["text"],
                    file_wave=str(it["out"]), seed=0)

    _run_clone(items, infer_one)


def synth_cosyvoice3(items):
    import torchaudio
    with _chdir(cfg.SOTA / "CosyVoice3"):
        # the `cosyvoice` package lives at the CosyVoice3 root; cwd is NOT on sys.path for a
        # script launched from elsewhere, so add both the root and the vendored Matcha-TTS.
        sys.path.insert(0, str(cfg.SOTA / "CosyVoice3"))
        sys.path.insert(0, str(cfg.SOTA / "CosyVoice3" / "third_party" / "Matcha-TTS"))
        from cosyvoice.cli.cosyvoice import AutoModel
        cosy = AutoModel(model_dir="pretrained_models/Fun-CosyVoice3-0.5B")
        prompts = cfg.load_prompts()
        enroll = {}
        for it in items:
            spk = it["speaker"]
            if spk not in enroll:
                enroll[spk] = cfg.enrollment(spk, prompts)
            ref_wav, ref_text = enroll[spk]
            full_prompt = "You are a helpful assistant.<|endofprompt|>" + ref_text
            _mkout(it)
            for j in cosy.inference_zero_shot(it["text"], full_prompt, str(ref_wav), stream=False):
                torchaudio.save(str(it["out"]), j["tts_speech"], cosy.sample_rate)
                break   # first (non-streaming) chunk is the full utterance
            print("saved", it["out"])


# --------------------------------------------------------------------------------------
# VITS baseline: VCTK speaker-id only -> identical audio for every L2 speaker
# --------------------------------------------------------------------------------------
def synth_vits(items):
    import torch
    from scipy.io.wavfile import write
    with _chdir(cfg.SOTA / "vits"):
        sys.path.insert(0, ".")
        import utils, commons
        from models import SynthesizerTrn
        from text.symbols import symbols
        from text import text_to_sequence

        hps = utils.get_hparams_from_file("./configs/vctk_base.json")
        net_g = SynthesizerTrn(
            len(symbols), hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers, **hps.model)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        net_g = net_g.to(dev).eval()
        utils.load_checkpoint("pretrained_vctk.pth", net_g, None)

        def get_text(text):
            seq = text_to_sequence(text, hps.data.text_cleaners)
            if hps.data.add_blank:
                seq = commons.intersperse(seq, 0)
            return torch.LongTensor(seq)

        sid = torch.LongTensor([cfg.VITS_SID]).to(dev)
        sr = hps.data.sampling_rate
        # accent/speaker are ignored by this checkpoint -> render once per utterance.
        rendered = {}
        for it in items:
            uid = it["utt_id"]
            if uid not in rendered:
                x = get_text(it["text"]).to(dev).unsqueeze(0)
                xl = torch.LongTensor([x.size(1)]).to(dev)
                with torch.no_grad():
                    rendered[uid] = net_g.infer(
                        x, xl, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                        length_scale=1)[0][0, 0].cpu().float().numpy()
            _mkout(it)
            write(str(it["out"]), sr, rendered[uid])
            print("saved", it["out"])


# --------------------------------------------------------------------------------------
# Parler baseline: NL accent description -> audio depends on (accent, gender, utterance)
# --------------------------------------------------------------------------------------
def synth_parler(items):
    import torch
    import soundfile as sf
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(dev)
    tok = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

    def describe(accent, gender):
        g = "female" if gender == "F" else "male"
        return (f"A {g} speaker with a {accent} accent delivers their words clearly at a "
                f"moderate speed and pitch. The recording is of very high quality, with the "
                f"speaker's voice sounding clear and very close up.")

    rendered = {}   # (accent, gender, utt_id) -> (audio, sr)
    for it in items:
        gender = cfg.SPEAKER_GENDER[it["speaker"]]
        key = (it["accent"], gender, it["utt_id"])
        if key not in rendered:
            di = tok(describe(it["accent"], gender), return_tensors="pt").to(dev)
            pi = tok(it["text"], return_tensors="pt").to(dev)
            gen = model.generate(
                input_ids=di.input_ids, attention_mask=di.attention_mask,
                prompt_input_ids=pi.input_ids, prompt_attention_mask=pi.attention_mask)
            rendered[key] = (gen.cpu().numpy().squeeze(), model.config.sampling_rate)
        audio, sr = rendered[key]
        _mkout(it)
        sf.write(str(it["out"]), audio, sr)
        print("saved", it["out"])


ADAPTERS = {
    "xtts": synth_xtts, "f5tts": synth_f5tts, "cosyvoice3": synth_cosyvoice3,
    "vits": synth_vits, "parler": synth_parler,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(ADAPTERS))
    ap.add_argument("--ref-utt", default=None,
                    help="override the per-speaker enrollment clip (cloning models)")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N grid cells (smoke test)")
    ap.add_argument("--overwrite", action="store_true",
                    help="regenerate clips that already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be written; loads no model libraries")
    args = ap.parse_args()

    if args.ref_utt:
        cfg.REFERENCE_UTT = args.ref_utt
    family = cfg.MODEL_FAMILY[args.model]
    if family == "clone" and cfg.REFERENCE_UTT in cfg.UTTERANCES:
        sys.exit(f"ERROR: reference utt {cfg.REFERENCE_UTT!r} is one of the 10 eval "
                 f"utterances -> cloning leakage. Pick a held-out utterance.")

    items = []
    for cell in cfg.grid():
        cell["out"] = cfg.synth_path(args.model, cell["accent"], cell["speaker"], cell["utt_id"])
        items.append(cell)
    if args.limit:
        items = items[:args.limit]
    todo = [it for it in items if args.overwrite or not it["out"].exists()]

    print(f"model={args.model}  family={family}  cells={len(items)}  "
          f"to-generate={len(todo)}  skip-existing={len(items) - len(todo)}")
    if family == "clone":
        print(f"reference (per-speaker, held-out): {cfg.REFERENCE_UTT}")
    elif family == "speaker_id":
        print(f"VITS VCTK sid={cfg.VITS_SID} (native-accent baseline; same audio per utt)")
    elif family == "description":
        print("Parler NL-description baseline (audio keyed by accent+gender+utt)")

    if args.dry_run:
        for it in todo[:12]:
            print("  would write", it["out"])
        if len(todo) > 12:
            print(f"  ... +{len(todo) - 12} more")
        return
    if not todo:
        print("nothing to do.")
        return

    ADAPTERS[args.model](todo)
    print("done.")


if __name__ == "__main__":
    main()

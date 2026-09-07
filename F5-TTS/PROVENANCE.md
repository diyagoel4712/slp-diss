# Provenance

The LoRA-capable F5-TTS fork used by **AccentVector** (task-vector accent control).

- **Origin:** the `f5_tts_lora/` subdirectory of
  <https://github.com/the-bird-F/Expressive-Vectors>.
- **Upstream base commit:** `84a811ebf532921f4996a85ac21160a9254ac39c`
  (`master`, "simplify infer_cli"). Verified 2026-09-05: still the tip of
  `origin/master`, 0 commits since.
- **Second-order upstream:** Expressive-Vectors is itself a fork of
  <https://github.com/SWivid/F5-TTS> that adds LoRA fine-tuning
  (`src/f5_tts/configs/F5TTS_v1_LoRA.yaml`, `use_lora` / `lora_rank`) —
  machinery that stock F5-TTS does not ship.
- **License:** MIT (see `LICENSE`), inherited from both upstreams.

## Version-control status — "Option B" (fork)

This directory **is its own git repository**, and its history is real: the
`f5_tts_lora/` subtree of Expressive-Vectors was extracted to the repository
root with

```
git clone https://github.com/the-bird-F/Expressive-Vectors.git ev
cd ev && git checkout 84a811ebf532921f4996a85ac21160a9254ac39c
git subtree split --prefix=f5_tts_lora -b av-base     # -> e1da0f9…
```

which yields the 5 upstream commits that touch `f5_tts_lora/`, rooted at
`e1da0f9140f6bfe1166924ef740b3b49ffb9de21` and laid out exactly as this
directory. The AccentVector patches sit on top of that base as ordinary
commits, so `git log` and `git diff e1da0f9` give the full, exact delta from
upstream. The split is deterministic — re-running the commands above reproduces
`e1da0f9` byte-for-byte.

Rooting at the subtree (rather than forking the whole Expressive-Vectors repo)
keeps the package at `F5-TTS/src/f5_tts/…`, which is the path every training and
inference script in `slp-diss` and on Eddie already uses.

### AccentVector patches (delta from `e1da0f9`)

Modified:

- `src/f5_tts/model/trainer.py` — LoRA-only snapshots on a `snapshot_per_updates`
  cadence and live accent-vector geometry logging (RQ1 trajectory, RQ5 geometry),
  plus optional multilingual-ASR (faster-whisper) + jiwer WER on logged samples.
- `src/f5_tts/model/backbones/dit.py` — guard the `lora_map` lookup so a
  single-LoRA-per-run config (`lora_feature_dim=None`) does not `KeyError` on
  data that still carries a per-sample `lora_idx`.
- `src/f5_tts/train/finetune_cli.py` — wire `sample_per_updates`, `track_wer`,
  `asr_language`, `asr_model_name`, `snapshot_per_updates`, `num_workers`;
  tolerate an empty `vocoder.local_path`.
- `src/f5_tts/train/train.py` — the same sample/WER config wiring.

Added:

- `src/f5_tts/configs/F5TTS_v1_LoRA_accent.yaml` — per-accent finetune config.
- `PROVENANCE.md` (this file), `.gitignore`.

Not tracked: `ckpts/`, `data/` (local weights and corpora).

### Consuming this fork from `slp-diss`

`slp-diss` references this repository as a git submodule at `F5-TTS/`, so a
reproduction is:

```
git clone --recursive <slp-diss url>
```

MIT attribution (`LICENSE` + this file) ships with the redistributed code.

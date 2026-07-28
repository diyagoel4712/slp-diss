# Provenance

The LoRA-capable F5-TTS fork used by **AccentVector** (task-vector accent control).

- **Origin:** the `f5_tts_lora/` subdirectory of
  <https://github.com/the-bird-F/Expressive-Vectors>, cloned locally.
- **Upstream:** a fork of <https://github.com/SWivid/F5-TTS> that adds LoRA
  fine-tuning (`src/f5_tts/configs/F5TTS_v1_LoRA.yaml`, `use_lora` / `lora_rank`)
  — machinery that stock F5-TTS does not ship.
- **License:** MIT (see `LICENSE`).
- **How tracked:** relocated to the dissertation repo root as `F5-TTS/`, but
  **gitignored — not vendored or committed** (it carries its own `.git` and MIT
  license). It is therefore NOT version-controlled inside this repo; to
  reconstruct it, clone Expressive-Vectors and move its `f5_tts_lora/`
  subdirectory to `F5-TTS/` (see `AccentVector/README.md` → Setup). Record the
  exact upstream commit here if you need the results reproducible against a moving
  upstream.

## Version-control status — interim "Option A" (force-add)

As of 2026-07-27 this directory is **not** a git clone (the `.git` mentioned above
was stripped on relocation). To keep the accent-vector / RQ6 patches under version
control *without* committing the ~54 MB vendored tree, the patched source files are
**force-added** into the parent `slp-diss` repo. They live under the wholesale
`F5-TTS/` rule in `.gitignore`, so `git add -f` is required the first time each is
committed; afterwards they track normally (`.gitignore` only masks untracked files).

Files tracked this way:

- `src/f5_tts/model/trainer.py`      — RQ6 snapshots/geometry + multilingual-ASR WER on samples
- `src/f5_tts/model/backbones/dit.py` — single-LoRA-per-run `lora_map` guard fix
- `src/f5_tts/train/finetune_cli.py` — config wiring
- `src/f5_tts/train/train.py`        — config wiring
- `src/f5_tts/configs/F5TTS_v1_LoRA_accent.yaml` — per-accent finetune config
- `PROVENANCE.md`, `LICENSE`         — provenance + MIT attribution

**Upstream base:** Expressive-Vectors `master` @
`84a811ebf532921f4996a85ac21160a9254ac39c` (its `f5_tts_lora/` subdir). Verified
2026-07-27: the local copy differs from this commit only in the files listed above,
consistent with it being the clone base.

### TODO before any public / published release

Option A commits only the patched files, **not** the surrounding F5-TTS package, so a
fresh clone of `slp-diss` alone is **not runnable or reproducible**, and it is a weak
basis for a public artifact. Before open-sourcing the repo / submitting the thesis
artifact / releasing paper code, migrate F5-TTS to a proper fork ("Option B"):

1. Fork Expressive-Vectors (or upstream `SWivid/F5-TTS` + reapply the LoRA layer),
   commit these patches there with real history, and pin the base commit above.
2. Reference that fork from `slp-diss` as a git **submodule** or a documented
   `git clone` step so reproduction is `git clone --recursive`.
3. Confirm MIT attribution (LICENSE + this file) ships with the redistributed code.

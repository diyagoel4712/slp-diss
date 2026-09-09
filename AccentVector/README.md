# Accent Vector on F5-TTS

A port of **Accent Vector** (Lertpetchpun et al., 2026 — controllable accent
manipulation via task vectors) to the **F5-TTS** backbone, adapted from
[Expressive-Vectors](https://github.com/the-bird-F/Expressive-Vectors). The
method is backbone-agnostic:

```
tau_accent = theta_ft - theta_pre               # extract  (paper Eq. 1-3)
theta      = theta_pre + alpha * tau_accent      # scale    (paper Eq. 4)
theta      = theta_pre + sum_i a_i * tau_accent(i)  # mix    (paper Eq. 5-6)
```

Fine-tune F5-TTS on native speech of a target accent/language, take the
checkpoint difference as the *accent vector*, then scale it (accent strength) or add several together (mixed accents) and merge back into the base model before inference. No accented-English training data is required.

## How this differs from the paper

The paper uses XTTS-v2; we use F5-TTS. Two consequences:

1. **No language-ID token.** XTTS pins a `[lang]` token to English so the LoRA
   delta acts as an *accent* shift, not a language switch. F5-TTS has no such
   token — content language is set purely by the `gen_text` at inference. So our
   recipe is: fine-tune on target-language audio+transcript (the delta captures
   that language's acoustics/prosody), then at inference feed **English** text so
   content stays English while the merged delta pushes the accent. Cleaner on F5
   (no competing token), but *not* the paper's exact procedure.

2. **LoRA fine-tuning, always.** The task-vector identity `tau = theta_ft - theta_pre`
   holds for any fine-tune, but this project only ever does LoRA — the paper's own
   recipe (Eq. 3: `tau = theta_LoRA`, rank 16, all linear layers, lr 3e-5). The accent
   vector *is* the LoRA branch, scaled natively by `lora_alpha` with **no checkpoint
   merge**: `infer_accent` builds the base+LoRA model **once** and rescales the branch
   per alpha in place (`src/accent_vector/lora_model.py`), feeding `score_sweep` /
   `score_prosody` and the figures notebook. Vectors are ~30 MB rather than a full
   checkpoint, and the geometry is cleaner.

   Rank, learning rate and schedule live in the fork's `F5TTS_v1_LoRA_accent.yaml`
   and can be overridden per run on the Hydra CLI, e.g.

   ```bash
   qsub scripts/finetune/finetune_wrapper_eddie.sh model.arch.lora_rank=32 optim.learning_rate=1e-4
   ```

   `scripts/lib/patch_f5_tts_fork.py` warns when the installed config differs from the
   recipe the write-up reports, so a sweep is never mistaken for the headline run.

   > A LoRA snapshot's delta lives in adapter keys that are absent from the base
   > checkpoint, so a full-weight diff of it is empty by construction — which is why
   > there is no merged/full-rank path here at all. `extract_vector extract-lora`
   > slices the branch out of a training checkpoint when no `lora_<step>.pt` snapshot
   > was written; `infer_accent` then scales it.

3. **Native-language reference.** F5 clones the reference clip,
   so its accent feeds the output. Following the paper's cloning setup, we provide
   a **native-language (L1) reference of the target accent** at inference, held
   **fixed per speaker across the alpha sweep** so the vector is the only thing that
   varies within a sweep. The sweep then runs between two exact anchors: **alpha=0
   is the pretrained model** (theta_pre) cloning the accent from the reference
   alone, no fine-tuning; **alpha=1 is the fully fine-tuned model** (theta_pre +
   tau), the full accent-vector impact. So the sweep measures the fine-tuning's
   contribution as accent strength climbs from the base cloning level to the full
   fine-tune, while speaker similarity should stay high.

## Setup (training needs a CUDA GPU)

Training is infeasible on CPU/Mac (F5-TTS pins `torch==2.4.0+cu124`; the paper
used an A40). **Evaluation runs on the Mac.**

F5-TTS lives at the repo-root **`F5-TTS/`** (the LoRA-capable fork). The scripts
default `F5_ROOT` to `../F5-TTS`; override it to point elsewhere. The fork is
required: it supplies the LoRA machinery and the `F5TTS_v1_LoRA_accent` config —
**stock `SWivid/F5-TTS` has no LoRA at all**.

> **Provenance of `F5-TTS/` (git submodule).** It is a fork of the `f5_tts_lora`
> subdirectory of <https://github.com/the-bird-F/Expressive-Vectors>, itself a
> fork of <https://github.com/SWivid/F5-TTS> that adds LoRA (MIT licensed). The
> subtree was extracted to the repo root with `git subtree split`, pinned at
> upstream `84a811e` (split base `e1da0f9`), with the AccentVector patches
> committed on top. Clone this repo with `--recursive`, or run
> `git submodule update --init` in an existing checkout. See
> `F5-TTS/PROVENANCE.md` for the exact delta from upstream.

```bash
conda create -n f5-tts python=3.11 -y && conda activate f5-tts
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124
cd F5-TTS && pip install -e . && cd ..
# put the base checkpoint at F5-TTS/ckpts/F5TTS_v1_Base/model_1250000.pt
# and the pretrained data/vocab.txt under F5-TTS/data/
```

Every stage puts `$F5_ROOT/src` and this package on `PYTHONPATH` for you.

## Pipeline 

```bash
# 1. Prepare the corpus -> F5 Arrow dataset  (metadata.csv -> <name>_pinyin/)
python main.py data prepare --metadata data/finetuning_data/dutch/metadata.csv \
    --audio-root /path/to/clips --out-dir $F5_ROOT/data/dutch_pinyin

# 2. LoRA fine-tune -> lora_<step>.pt snapshots  (GPU; Eddie: qsub finetune_wrapper_eddie.sh)
ACCENT_NAME=dutch bash scripts/finetune/finetune.sh

# 3. Alpha sweep over held-out English transcripts, fixed L1 reference  (GPU)
#    alpha=0 = base model cloning the reference; alpha=1 = the trained LoRA strength
VECTOR=<run>/ckpts/snapshots/lora_60000.pt CONFIG=<run>/config.yaml VOCAB=<run>/vocab.txt \
  REF_AUDIO=data/prompts/dutch/dutch_f.wav bash scripts/infer/infer_sweep.sh

# 4. Score the sweep -> rq1.csv / rq3.csv  (Mac; Eddie: submit_eval_grid.sh)
python -m accent_vector.score_sweep --sweep-dir <sweep> --lid --out-csv <sweep>/rq1.csv
python -m accent_vector.score_prosody --sweep-dir <sweep> --natural-ref <gt> --out-csv <sweep>/rq3.csv
```

**Go/no-go:** accent similarity (`accent_cs`) should rise with alpha while speaker
similarity stays high (≈0.9 in the paper).

### Or drive stages directly

```bash
# slice the LoRA vector out of a training checkpoint (only if no lora_<step>.pt exists)
python main.py vector extract-lora --checkpoint <run>/ckpts/model_last.pt --out vectors/dutch.pt

# the alpha sweep
python main.py infer --pretrained .../model_1250000.pt --lora-vector vectors/dutch.pt \
    --config <run>/config.yaml --vocab <run>/vocab.txt \
    --alphas 0,0.25,0.5,0.75,1.0 \
    --ref-audio data/prompts/dutch/dutch_f.wav --ref-text "..." \
    --transcripts data/transcripts/dutch/dutch_f_eval.txt \
    --out-dir results/per-accent/dutch
```

## Mixed accents (paper Eq. 5-6) — not implemented

The paper composes accents as `theta_pre + sum_i a_i * tau_i`. That was previously
done by merging full-weight vectors into a checkpoint; with LoRA-only fine-tuning
there is no merge step, and `lora_model` overlays exactly one branch at a time, so
**mixing is currently unavailable**. Implementing it means overlaying several LoRA
state dicts with independent `lora_alpha` values (the fork's `lora_idx` /
`lora_mapping` multi-branch support is the obvious hook), not reviving a full-weight
merge.

## Data & later phases

- **Train (per accent):** ~100 h of **native-language (L1)** speech — one dataset or
  several combined into a single `audio_file|text` CSV (use **absolute** audio paths so
  one `--audio-root` covers every source), then `data_preprocess prepare` → fine-tune →
  sweep. 100 h is ample for a rank-16 LoRA vector.
- **Test:** a **bilingual** corpus (each speaker recorded in their L1 **and** in English)
  is ideal — the L1 clips are the cloning references, the natural English recordings are
  the target-accent clips for `cs_accent`/PPG-KL/F0, same speaker for both. Code-switching
  data works but must be segmented into clean L1 vs English spans. Keep test speakers
  **disjoint** from the fine-tuning set.
- **Several speakers per accent:** the submit scripts (`scripts/infer/submit_<accent>_ckpt_grid.sh`)
  sweep each speaker into `results/per-accent/<accent>/<tag>/<ref_kind>/<speaker>/`. Score each
  with `score_sweep` / `score_prosody`; the figures notebook pools the m/f speakers per accent.
- **Non-Latin transcripts:** the F5 base vocab covers Latin + pinyin only, so Hindi/Arabic/
  Korean L1 transcripts won't tokenize — romanize them or extend the vocab **before**
  fine-tuning.


## Layout

```
src/accent_vector/       the importable package -- src layout, so PYTHONPATH=src
  data_preprocess.py     corpus -> metadata.csv -> F5 Arrow dataset
  extract_vector.py      slice the LoRA vector out of a checkpoint + layer masking
  lora_model.py          base+LoRA build; scales the branch in place (Eq. 4)
  infer_accent.py        alpha-sweep / single-ckpt inference
  score_sweep.py         -> rq1.csv: accent, identity, WER, P(English), onsets
  score_prosody.py       -> rq3.csv: segmental (PPG-KL) + suprasegmental
  shared.py              eval-suite bridge, sweep IO, geometry, threshold onset
  sample_checkpoints.py  synthesise a fixed prompt at every LoRA snapshot
  experiments/           unfinished analyses, kept for the write-up (see its
                         __init__.py: none of them has produced output)
scripts/                 finetune / extract / infer / evaluate wrappers and the
                         Eddie array jobs (these export PYTHONPATH=$ACCENT_DIR/src)
notebooks/               dissertation_figures.ipynb -- every figure, by RQ
data/                    all inputs (git-ignored; see .gitignore)
  finetuning_data/       per-accent fine-tuning metadata.csv
  transcripts/           held-out English eval transcripts
  prompts/               per-accent L1 reference clips + GAE/ neutral controls
  ground_truth_refs/     natural target-accent clips (accent CS + RQ3 targets)
main.py                  dispatcher (data/vector/infer/evaluate/score-sweep/
                         score-prosody); adds src/ to sys.path itself
```

`vectors/`, `results/`, `exps/` and any F5 checkpoints are generated artifacts and
are git-ignored, as is `data/` apart from the eval assets re-included in
`.gitignore` (`data/transcripts/`, `data/prompts/`, `data/ground_truth_refs/`).

## Typical run order

```bash
# after A0 produces the LoRA snapshots: submit the checkpoint x alpha grid.
# (accent_vector.experiments.grid was the standalone driver for this and is gone;
#  the Eddie array does the job -- see scripts/infer/submit_<accent>_ckpt_grid.sh)
bash scripts/infer/submit_dutch_ckpt_grid.sh      # A1 -> results/per-accent/<accent>/<ref>/<spk>/audio/step_<n>/alpha_<a>/

# Core scoring: score each speaker with ITS own L1 reference + natural clips, then pool
for s in results/per-accent/indian/*/; do sp=$(basename "$s")
  python -m accent_vector.score_sweep --sweep-dir "$s" \
      --transcripts data/transcripts/eval_transcripts.txt --ref-wav data/prompts/$acc/$sp.wav \
      --accent-ref data/ground_truth_refs/$acc/$g --lid --out-csv "$s/rq1.csv"
  python -m accent_vector.score_prosody --sweep-dir "$s" \
      --natural-ref data/ground_truth_refs/$acc/$g --out-csv "$s/rq3.csv"
done
python -m accent_vector.experiments.aggregate --accent-dir results/per-accent/indian --csv-name rq1.csv --out-dir results/per-accent/indian
python -m accent_vector.experiments.aggregate --accent-dir results/per-accent/indian --csv-name rq3.csv --out-dir results/per-accent/indian
# (aggregate is unrun -- see experiments/__init__.py)

python -m accent_vector.experiments.rq3_layers --vector vectors/indian.pt \
    --out-csv results/per-accent/indian/rq3_layers.csv          # vector-only; UNRUN, see experiments/__init__.py
python -m accent_vector.experiments.rq5_geometry --vector indian=vectors/indian.pt \
    --vector spanish=vectors/spanish.pt --synth indian=results/per-accent/indian/p1/alpha_1.0 \
    --synth spanish=results/per-accent/spanish/s1/alpha_1.0 --out-dir results/geometry   # UNRUN
python -m accent_vector.experiments.rq2_temporal \
    --ckpt-dir exps/F5TTS_v1_LoRA_indian/<run>/ckpts/snapshots \
    --out-csv results/per-accent/indian/temporal.csv            # UNRUN
```

### Layer-masked accent — "accent without the language"

On the LoRA track the mask is native: `set_lora_alpha` zeroes the un-selected
branches, so `infer_accent` takes `--include-layers` /
`--exclude-layers` (substrings over LoRA submodule names — `attn`, `ff`, `conv`,
`text_embed`, `input_embed`, `lora_proj_out`, or a block index like `.5.`). Excluding
the content/language path is the direct attack on the α=1 gibberish:

```bash
# scale accent-carrying layers only; leave the text/content path at theta_pre
python -m accent_vector.infer_accent \
    --pretrained ckpts/F5TTS_v1_Base/model_1250000.pt --lora-vector vectors/dutch.pt \
    --config <run>/config.yaml --vocab <run>/vocab.txt --alphas 0,0.5,1.0 \
    --exclude-layers text_embed --exclude-layers input_embed \
    --ref-audio data/prompts/GAE/gae_f.wav --ref-text "..." \
    --transcripts data/transcripts/eval_transcripts.txt --out-dir results/per-accent/dutch/masked
# then score results/per-accent/dutch/masked with rq1/rq3 as usual and compare to the unmasked sweep
```

### Checkpoint × alpha comparison — accent-vs-language over training

Needs `snapshot_per_updates` snapshots (or full `model_<step>.pt`) from A0. Render
each checkpoint's sweep, score each, then collate at matched α — ideally against the
**neutral** reference so an accent rise is the vector, not cloning:

```bash
# GPU: alpha sweep at several checkpoints -> results/per-accent/dutch/GAE/by_step/step_<step>/
STEP_INTERVAL=5000 REF_KINDS=GAE bash scripts/infer/submit_dutch_ckpt_grid.sh

# CPU: score each checkpoint, then compare matched-alpha across training
for s in results/per-accent/dutch/GAE/by_step/step_*/; do
  python -m accent_vector.score_sweep --sweep-dir "$s" \
    --transcripts data/transcripts/eval_transcripts.txt --ref-wav data/prompts/GAE/gae_f.wav \
    --accent-ref data/ground_truth_refs/dutch/female --lid --out-csv "$s/rq1.csv"
done
python -m accent_vector.experiments.rq2_behavioural \  # UNRUN
    --by-step-dir results/per-accent/dutch/GAE/by_step --out-dir results/per-accent/dutch/GAE/trajectory
```

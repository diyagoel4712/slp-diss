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
checkpoint difference as the *accent vector*, then scale it (accent strength) or
add several together (mixed accents) and merge back into the base model before
inference. No accented-English training data is required.

See [`PROPOSAL.md`](PROPOSAL.md) for the full design rationale and
phase sequencing.

## How this differs from the paper

The paper uses XTTS-v2; we use F5-TTS (best model in our own benchmark, already
set up in `Preliminary_test_results/f5-tts`). Two consequences:

1. **No language-ID token.** XTTS pins a `[lang]` token to English so the LoRA
   delta acts as an *accent* shift, not a language switch. F5-TTS has no such
   token — content language is set purely by the `gen_text` at inference. So our
   recipe is: fine-tune on target-language audio+transcript (the delta captures
   that language's acoustics/prosody), then at inference feed **English** text so
   content stays English while the merged delta pushes the accent. Cleaner on F5
   (no competing token), but *not* the paper's exact procedure.

2. **Two fine-tune tracks — LoRA is the paper-matching one.** The task-vector
   identity `tau = theta_ft - theta_pre` holds either way; the paper uses LoRA
   (Eq. 3: `tau = theta_LoRA`, rank 16, all linear layers, lr 3e-5) and the
   dissertation (PROPOSAL.md) follows it — smaller ~30 MB vectors, cleaner
   geometry. Two paths exist in `scripts/`:
   - **LoRA** (`finetune_lora.sh`, `F5TTS_v1_LoRA_accent` config): the accent
     vector *is* the LoRA branch, scaled natively by `lora_alpha` (no merge).
     The whole analysis pipeline runs on it — `infer_accent --lora` (and
     `grid --lora`) build the base+LoRA model **once** and rescale the branch per
     alpha in place (`accent_vector/lora_model.py`), feeding RQ1–RQ3, plus the RQ6
     trajectory tooling (`sample_checkpoints.py`, `rq_temporal --lora`,
     `viz_temporal`). This is the paper-matching, default track.
   - **Full fine-tune** (`finetune.sh`, `F5TTS_v1_Base`): the path
     Expressive-Vectors exercises end-to-end; `extract_vector` diffs the full
     checkpoint and `compose` merges `theta_pre + alpha*tau` for the alpha sweep.
     Use it when you want a full-weight vector rather than a LoRA branch.

   > Both tracks emit the identical `results/<accent>/alpha_<a>/utt####.wav`
   > layout, so `evaluate` and every `rq*` analysis read them the same way. Note a
   > LoRA snapshot cannot go through `extract`/`compose` (its delta lives in new
   > adapter keys absent from the base, so the diff is empty) — scale it natively
   > with `--lora` instead.

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
default `F5_ROOT` to `../F5-TTS`; override it to point elsewhere. For the LoRA
path (as compared to full finetuning) you need this fork's `F5TTS_v1_LoRA`
config; **stock `SWivid/F5-TTS` has no LoRA**, so only use it if you stick to the
full fine-tune.

> **Provenance of `F5-TTS/` (gitignored, not vendored).** It is the `f5_tts_lora`
> subdirectory of <https://github.com/the-bird-F/Expressive-Vectors>, itself a
> fork of <https://github.com/SWivid/F5-TTS> that adds LoRA (MIT licensed). To
> reconstruct it: clone Expressive-Vectors and move `f5_tts_lora` to `F5-TTS/`.
> (A copy of this note also lives in `F5-TTS/PROVENANCE.md`.)

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
# 1. Fine-tune F5-TTS on VCTK England speech  -> theta_ft   (GPU)
VCTK_ROOT=/data/VCTK-Corpus-0.92 bash scripts/finetune.sh

# 2. Extract the accent vector: tau = theta_ft - theta_pre  (Eq. 1-3)
FINETUNED=.../ckpts/british/model_60000.pt bash scripts/extract_vector.sh

# 3. Alpha sweep over held-out English transcripts, fixed native-language (L1) ref  (GPU)
#    alpha=0 = pretrained model cloning the reference; alpha=1 = fully fine-tuned
REF_AUDIO=refs/england.wav bash scripts/infer_sweep.sh

# 4. Score the sweep with the repo eval suite -> metrics.csv  (Mac)
bash scripts/evaluate.sh
```

**Go/no-go for the port:** in `metrics.csv`, accent similarity should rise
monotonically with alpha while speaker similarity stays high (≈0.9 in the paper).

### Or drive stages directly

```bash
python main.py vector extract  --pretrained .../model_1250000.pt --finetuned .../model_60000.pt --out vectors/british.pt
python main.py vector compose  --pretrained .../model_1250000.pt --vector vectors/british.pt --alpha 0.6 --out ckpts/british/a0.6.pt
python main.py infer    --pretrained .../model_1250000.pt --vector vectors/british.pt --alphas 0,0.2,0.4,0.6,0.8,1.0 \
                        --ref-audio refs/england.wav --ref-text "..." --transcripts transcripts/eval_transcripts.txt --out-dir results/british
python main.py evaluate --sweep-dir results/british --transcripts transcripts/eval_transcripts.txt --ref-wav refs/england.wav --out-csv results/british/metrics.csv
```

## Mixed accents (paper Eq. 5-6)

Compose two vectors, then synthesize the single merged checkpoint:

```bash
python main.py vector compose --pretrained .../model_1250000.pt \
    --vector vectors/spanish.pt --alpha 0.5 \
    --vector vectors/british.pt --alpha 0.5 \
    --out ckpts/mixed/spanish+british.pt
python main.py infer --ckpt ckpts/mixed/spanish+british.pt \
    --ref-audio refs/england.wav --ref-text "..." \
    --transcripts transcripts/eval_transcripts.txt --out-dir results/spanish+british
```

## Later phases

- **For other accents**: supply your own `audio_file|text` CSV,
  run `data_preprocess prepare`, then the same fine-tune → extract → sweep flow.
- **For non-Latin transcripts**: the F5 base vocab covers
  Latin + pinyin only, so native transcripts won't tokenize. Romanize the
  transcripts or extend the vocab first.

## Layout

```
accent_vector/
  data_preprocess.py   VCTK -> metadata.csv -> F5 Arrow dataset
  extract_vector.py    task-vector extract (Eq. 1-3) + arithmetic (Eq. 4-6)
  infer_accent.py      alpha-sweep / single-ckpt inference, fixed native-L1 ref
  evaluate.py          scores a sweep via the Evaluation/ eval suite
  sample_checkpoints.py  RQ6: synthesise a fixed prompt at every LoRA snapshot
  experiments/         dissertation RQ harness (see EXPERIMENTS.md);
                       grid, rq1_reproduction, rq2_geometry, rq3_decomposition,
                       rq3_layers, rq_temporal, viz_temporal, common
scripts/               finetune(.sh) / finetune_lora(.sh) / extract / infer /
                       evaluate wrappers
transcripts/           held-out English eval transcripts (10 CMU ARCTIC sents)
main.py                unified dispatcher over the core stages (data/vector/infer/evaluate)
```

`data/`, `vectors/`, `results/`, `refs/`, and any F5 checkpoints are generated
artifacts and are git-ignored.

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

See [`ADAPTATION_PLAN.md`](ADAPTATION_PLAN.md) for the full design rationale and
phase sequencing.

## How this differs from the paper

The paper uses XTTS-v2; we use F5-TTS (best model in our own benchmark, already
set up in `SOTA_models_experiments/f5-tts`). Two consequences, both verified
against the installed F5-TTS:

1. **No language-ID token.** XTTS pins a `[lang]` token to English so the LoRA
   delta acts as an *accent* shift, not a language switch. F5-TTS has no such
   token — content language is set purely by the `gen_text` at inference. So our
   recipe is: fine-tune on target-language audio+transcript (the delta captures
   that language's acoustics/prosody), then at inference feed **English** text so
   content stays English while the merged delta pushes the accent. Cleaner on F5
   (no competing token), but *not* the paper's exact procedure.

2. **Full fine-tune, not LoRA (by default).** The task-vector identity
   `tau = theta_ft - theta_pre` holds for both; the paper's LoRA is an
   efficiency/regularization choice (Eq. 3: `tau = theta_LoRA`). We default to
   the full fine-tune that Expressive-Vectors exercises end-to-end. To match the
   paper's LoRA (rank 16, all linear layers, lr 3e-5), switch `finetune.sh` to
   the `F5TTS_v1_LoRA` config — but first confirm the checkpoint stores **merged**
   weights, otherwise the delta lives in new adapter keys absent from the base
   model and `extract` will (correctly) report an empty vector.

3. **Fixed neutral reference.** F5 clones the reference clip's own accent, which
   would confound the alpha sweep. We hold the reference fixed and neutral across
   all alpha so the vector is the only thing varying.

## Setup (training needs a CUDA GPU)

Training is infeasible on CPU/Mac (F5-TTS pins `torch==2.4.0+cu124`; the paper
used an A40). **Evaluation runs on the Mac.**

F5-TTS lives at the repo-root **`F5-TTS/`** (the LoRA-capable fork). The scripts
default `F5_ROOT` to `../F5-TTS`; override it to point elsewhere. For the LoRA
path (recommended — see deviation #2) you need this fork's `F5TTS_v1_LoRA`
config; **stock `SWivid/F5-TTS` has no LoRA**, so only use it if you stick to the
full fine-tune.

```bash
conda create -n f5-tts python=3.11 -y && conda activate f5-tts
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 \
  --extra-index-url https://download.pytorch.org/whl/cu124
cd F5-TTS && pip install -e . && cd ..
# put the base checkpoint at F5-TTS/ckpts/F5TTS_v1_Base/model_1250000.pt
# and the pretrained data/vocab.txt under F5-TTS/data/
```

Every stage puts `$F5_ROOT/src` and this package on `PYTHONPATH` for you.

## Pipeline (Phase A: British-accented English)

```bash
# 1. Fine-tune F5-TTS on VCTK England speech  -> theta_ft   (GPU)
VCTK_ROOT=/data/VCTK-Corpus-0.92 bash scripts/finetune.sh

# 2. Extract the accent vector: tau = theta_ft - theta_pre  (Eq. 1-3)
FINETUNED=.../ckpts/british/model_60000.pt bash scripts/extract_vector.sh

# 3. Alpha sweep over held-out English transcripts, fixed neutral ref  (GPU)
REF_AUDIO=refs/neutral.wav bash scripts/infer_sweep.sh

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
                        --ref-audio refs/neutral.wav --ref-text "..." --transcripts transcripts/eval_transcripts.txt --out-dir results/british
python main.py evaluate --sweep-dir results/british --transcripts transcripts/eval_transcripts.txt --ref-wav refs/neutral.wav --out-csv results/british/metrics.csv
```

## Mixed accents (paper Eq. 5-6)

Compose two vectors, then synthesize the single merged checkpoint:

```bash
python main.py vector compose --pretrained .../model_1250000.pt \
    --vector vectors/spanish.pt --alpha 0.5 \
    --vector vectors/british.pt --alpha 0.5 \
    --out ckpts/mixed/spanish+british.pt
python main.py infer --ckpt ckpts/mixed/spanish+british.pt \
    --ref-audio refs/neutral.wav --ref-text "..." \
    --transcripts transcripts/eval_transcripts.txt --out-dir results/spanish+british
```

## Later phases

- **Phase B — Vietnamese** (Latin script): supply your own `audio_file|text` CSV,
  run `data_preprocess prepare`, then the same fine-tune → extract → sweep flow.
- **Phase C — Hindi / Arabic / Korean** (non-Latin): the F5 base vocab covers
  Latin + pinyin only, so native transcripts won't tokenize. Romanize the
  transcripts or extend the vocab first (ADAPTATION_PLAN.md gotcha #2). Hindi is
  worth the effort because GenAID actually classifies Indian → a validated
  accent-ID number, not just embedding cosine.

## Layout

```
accent_vector/
  data_preprocess.py   VCTK -> metadata.csv -> F5 Arrow dataset
  extract_vector.py    task-vector extract (Eq. 1-3) + arithmetic (Eq. 4-6)
  infer_accent.py      alpha-sweep / single-ckpt inference, fixed neutral ref
  evaluate.py          scores a sweep via SOTA_models_experiments eval suite
scripts/               finetune / extract / infer / evaluate wrappers
transcripts/           held-out English eval transcripts (10 CMU ARCTIC sents)
main.py                unified dispatcher over the stages
```

`data/`, `vectors/`, `results/`, `refs/`, and any F5 checkpoints are generated
artifacts and are git-ignored.

# SOTA models — accent-generation benchmark (archived)

Archived from `main` on 2026-09-07. This is the **script generation** of the benchmark;
its earlier notebook generation is in `SOTA_models_experiments/` on this same branch
(`run_evaluation.ipynb`, `vits.ipynb`, `f5-tts/synthesise.ipynb`,
`parler-tts/synthesise.ipynb`, `select_utterances.py`).

Nothing in the dissertation pipeline on `main` depends on these files. They are kept
because they produced the baseline numbers in `evaluation_results.csv`.

## What it did

Run evaluation across a range of open-source models to assess their accent-generation
capabilities, and thereby define the problem space that current models face when
generating text in different accents.

Each model is prompted with 4 utterances, each to be synthesised in 4 different accents
by 5 different speakers. Each speaker-utterance combination is evaluated against a
reference speech sample.

All models are zero-shot. Some (XTTS-2, VITS, F5) require reference speech; others
(CosyVoice3, Parler-TTS) require natural-language descriptions of the desired speaker
characteristics. The accents are Arabic, Indian, Vietnamese and Korean.

## The files

| file | role |
|---|---|
| `eval_config.py` | the grid: utterances, accents, speakers, models, per-speaker references. Imported by the three below. |
| `synthesis_driver.py` | generation. Run **once per model, inside that model's own env** — the models have conflicting dependencies. Writes `SOTA_models_experiments/<model>/outputs/<model>/<accent>/<speaker>/<utt_id>.wav`. |
| `run_eval.py` | scoring. One row per clip that has both a synthesised wav and a reference → `evaluation_results.csv`. |
| `visualize_results.py` | summaries + figures over that CSV. |
| `run_pipeline.sh` | drives `run_eval.py` then `visualize_results.py`. |
| `arctic.data` | CMU ARCTIC id → prompt text (read by `eval_config.PROMPTS`). |
| `evaluation_results.csv` | **the result**: 636 scored clips. On `main` this file was untracked and gitignored under `Preliminary_test_results/`; it is committed here at the path `run_eval.py` writes it to. |
| `evaluation_functions.py` | the metric implementations, copied from `main` so this archive runs standalone — see below. |

```bash
# per model, each in its own env
python synthesis_driver.py --model xtts        # coqui/XTTS env
python synthesis_driver.py --model f5tts       # f5-tts env
python synthesis_driver.py --model cosyvoice3
python synthesis_driver.py --model vits        # VCTK speaker-id, native-accent baseline
python synthesis_driver.py --model parler      # NL accent-description baseline

# then score + plot
bash run_pipeline.sh
```

## Relationship to `main`

`evaluation_functions.py` is **live on `main`** — it is the shared metric suite that
`AccentVector`'s `score_sweep.py` / `score_prosody.py` call through
`shared.load_eval()`. `run_eval.py` imports it, so a copy sits here to keep the archive
self-contained; the authoritative version is `Evaluation/evaluation_functions.py` on
`main`, and the two will drift.

Also staying on `main`, because the accent-vector eval needs them: `genaid_wrappers/`
(copied into the GenAID clone by `AccentVector/scripts/eddie_eval_setup.sh`),
`requirements-eval.txt` and `requirements-genaid.txt`. The environment and GenAID setup
instructions live in `Evaluation/README.md` on `main`.

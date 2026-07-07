# Experiment matrix

Runnable map of the dissertation plan (see [PROPOSAL.md](PROPOSAL.md)) onto the
code. **One synthesis grid feeds every analysis** — only Phase 0 needs a GPU;
all `rq*` modules run on the Mac against the grid's audio.

## Phase 0 — assets (GPU)

| ID | What | How |
|----|------|-----|
| A0 | One LoRA fine-tune → one vector per accent `{british, spanish, vietnamese, +1 distant}` | `scripts/finetune.sh` (set LoRA config) → `scripts/extract_vector.sh` |
| A1 | Synthesis grid: accent × α sweep, **fixed neutral** reference (isolates the vector) | `python -m accent_vector.experiments.grid --config grid.json` |
| A2 | Natural target-accent clips + GAE baseline clips (per accent) | data collection; endpoints for gap-closure / cs_accent |
| A3 | Reference-leakage ablation: same grid with **per-accent L1** reference | `grid --config grid.json --reference-mode matched` |

**Reference leakage (A3):** F5 clones the reference's own accent, so the matched-vs-fixed
gap at α=0 measures how much accent the reference alone supplies — separating it from the
vector's contribution. `matched` mode writes to `results/<accent>__matchedref/`; each accent's
L1 reference goes in the config's `references` block.

## Experiments (CPU / Mac, over the A1 grid)

| ID | RQ | Module | Output | Confirms hypothesis if |
|----|----|--------|--------|------------------------|
| E1.1 | RQ1 | `rq1_reproduction` | `rq1.csv` | accent ↑ monotonic with α (Spearman>0), spk-sim flat/high |
| E1.2 | RQ1 | `rq1_reproduction` (wer col) | `rq1.csv` | WER rises with α faster than paper's XTTS (leakage) |
| E1.3 | RQ1b | `rq1_reproduction --lid` (eng_lid col) | `rq1.csv` | P(English) falls with α — direct language drift, distinct from accent |
| E1.4 | RQ1b | `rq1_reproduction` (leak-onset in footer) | `rq1.csv` | leakage-onset α lower on F5 than XTTS (missing language anchor) |
| E2.1 | RQ2 | `rq2_geometry` | `weight_space_cosine.csv`, `..._mds.csv` | accents cluster by family in MDS |
| E2.2 | RQ2 | `rq2_geometry` (--synth) | `output_space_cosine.csv`, `rsa_mantel.txt` | Mantel r>0, p<0.05 but r<1 (imperfect) |
| E2.3 | RQ2 | `rq2_geometry` (within- vs cross-English) | matrices | corpus contributes measurable distance |
| E3.1 | RQ3 | `rq3_decomposition` (seg cols) | `rq3.csv` | PPG-KL-to-natural falls with α |
| E3.2 | RQ3 | `rq3_decomposition` (supra cols) | `rq3.csv` | F0/rhythm move little toward natural |
| E3.3 | RQ3 | `rq3_decomposition` (closure) | `rq3.csv` | seg_closure ≫ supra_closure_mean, widest for distant accent |
| E3.4 | RQ3 | `rq3_layers` | `rq3_layers.csv` | accent energy concentrates in identifiable modules/depth |
| E4.1 | RQ4* | `extract_vector compose --include` + `rq3_decomposition` | `rq3.csv` | up-weighting prosody-layers raises supra_closure |
| E4.2 | RQ4* | reference retrieval (stub) + `rq3` | — | matched reference raises supra transfer |
| E5.1 | RQ5 | `rq5_bias` (gender rows) | `rq5.csv` | metrics differ by gender |
| E5.2 | RQ5 | `rq5_bias` (wer_relative) | `rq5.csv` | relative WER < absolute WER (ASR bias share) |
| E5.3 | RQ5 | flowchart | figure | — (deliverable) |

`*` RQ4 is the stretch tier.

## Not yet wired (documented integration points)

- **LID probability** (E1.3) — `rq1_reproduction --lid` has the hook; it calls
  `evaluation_functions.predict_lid_english` if present. Wire that to
  VoxLingua107 (`speechbrain/lang-id-voxlingua107-ecapa`) in the isolated env to
  activate the eng_lid column and the LID-based leakage onset. Until then WER
  carries the leakage signal and the WER-based onset still reports.
- **XTTS token ablation** (RQ1b clean isolation) — out of scope here (needs XTTS
  re-stood-up); the F5-vs-XTTS onset gap confounds backbone with the missing
  language-ID token, so report it as evidence, not proof. See PROPOSAL.md RQ1b.
- **Forced-alignment rhythm** (%V, ΔC, nPVI) — `rq3` ships a voicing-based proxy
  from `extract_f0`; swap in MFA vowel/consonant intervals for the rigorous form.
- **Reference retrieval** (E4.2) — adapt `SOTA_models_experiments/select_utterances.py`.

## Typical run order

```bash
# after A0 produces vectors/*.pt and a grid.json listing them:
python -m accent_vector.experiments.grid --config grid.json                    # A1

python -m accent_vector.experiments.rq1_reproduction --sweep-dir results/british ...   # E1
python -m accent_vector.experiments.rq3_decomposition --sweep-dir results/british \
    --natural-ref data/england_natural --out-csv results/british/rq3.csv               # E3 (core)
python -m accent_vector.experiments.rq3_layers --vector vectors/british.pt \
    --out-csv results/british/rq3_layers.csv                                           # E3.4
python -m accent_vector.experiments.rq2_geometry --vector british=vectors/british.pt \
    --vector spanish=vectors/spanish.pt --synth british=results/british/alpha_1.0 \
    --synth spanish=results/spanish/alpha_1.0 --out-dir results/geometry               # E2
python -m accent_vector.experiments.rq5_bias --synth-dir results/british/alpha_1.0 \
    --transcripts transcripts/eval_transcripts.txt --vctk-root $VCTK ...               # E5
```

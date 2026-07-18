# Experiment matrix

Runnable map of the dissertation plan (see [PROPOSAL.md](PROPOSAL.md)) onto the
code. **One synthesis grid feeds every analysis** — only Phase 0 needs a GPU;
all `rq*` modules run on the Mac against the grid's audio.

## Phase 0 — assets (GPU)

| ID | What | How |
|----|------|-----|
| A0 | One LoRA fine-tune → one vector per accent `{british, spanish, vietnamese, +1 distant}` | `scripts/finetune_lora.sh` per accent; the vector is the final `lora_<step>.pt` snapshot (point the grid at it) |
| A1 | Synthesis grid: accent × α sweep, per-accent **native-language (L1) reference** held fixed across α | `python -m accent_vector.experiments.grid --config grid.json --lora` |
| A2 | Natural target-accent clips + GAE baseline clips (per accent) | data collection; endpoints for gap-closure / cs_accent |

(Full-fine-tune track instead: `scripts/finetune.sh` → `scripts/extract_vector.sh` → `grid` without `--lora`, which merges each alpha.)

**Sweep anchors (A1):** the reference is the accent's native-language (L1) clip (paper-faithful
cloning), fixed across the sweep. **α=0 = θ_pre** — the pretrained model cloning the accent from
the reference alone (no fine-tuning); **α=1 = θ_ft** — the fully fine-tuned model. So the sweep
measures the fine-tuning's contribution from the base cloning level up to the full fine-tune.
Each accent's reference goes in the config's `references` block.

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
| E6.1 | RQ6 | `rq_temporal` | `temporal.csv` | `cos(τ_t, τ_final)` converges before magnitude (direction learnable early) |

`*` RQ4 is the stretch tier. **E6.1 is Tier-1 only** (optimisation trajectory,
near-free): needs intermediate `model_<step>.pt` checkpoints saved during A0.
The data-efficiency variant (Tier 2/3: separate LoRAs on data fractions with
error bounds) is out of scope — step ≠ data amount.

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
- **Reference retrieval** (E4.2) — adapt `Evaluation/select_utterances.py`.

## Typical run order

```bash
# after A0 produces the LoRA snapshots and a grid.json listing them:
python -m accent_vector.experiments.grid --config grid.json --lora             # A1

python -m accent_vector.experiments.rq1_reproduction --sweep-dir results/british ...   # E1
python -m accent_vector.experiments.rq3_decomposition --sweep-dir results/british \
    --natural-ref data/england_natural --out-csv results/british/rq3.csv               # E3 (core)
python -m accent_vector.experiments.rq3_layers --vector vectors/british.pt \
    --out-csv results/british/rq3_layers.csv                                           # E3.4
python -m accent_vector.experiments.rq2_geometry --vector british=vectors/british.pt \
    --vector spanish=vectors/spanish.pt --synth british=results/british/alpha_1.0 \
    --synth spanish=results/spanish/alpha_1.0 --out-dir results/geometry               # E2
python -m accent_vector.experiments.rq_temporal --lora \
    --ckpt-dir exps/F5TTS_v1_LoRA_british/<run>/ckpts/snapshots \
    --out-csv results/british/temporal.csv                                             # E6
```

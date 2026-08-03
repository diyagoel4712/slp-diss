# Experiment matrix

Runnable map of the dissertation plan (see [PROPOSAL.md](PROPOSAL.md)) onto the
code. **One synthesis grid feeds every analysis** — only Phase 0 needs a GPU;
all `rq*` modules run on the Mac against the grid's audio.

## Phase 0 — assets (GPU)

| ID | What | How |
|----|------|-----|
| A0 | One LoRA fine-tune → one vector per accent `{british, spanish, vietnamese, +1 distant}` | `scripts/finetune_lora.sh` per accent; the vector is the final `lora_<step>.pt` snapshot (point the grid at it) |
| A1 | Synthesis grid: accent × **speaker** × α sweep, each speaker's **native-language (L1) reference** held fixed across α → `results/<accent>/<speaker>/alpha_<a>/` | `python -m accent_vector.experiments.grid --config grid.json --lora` |
| A2 | Natural target-accent clips + GAE baseline clips (per speaker) | data collection; endpoints for gap-closure / cs_accent |

(Full-fine-tune track instead: `scripts/finetune.sh` → `scripts/extract_vector.sh` → `grid` without `--lora`, which merges each alpha.)

**Sweep anchors (A1):** the reference is fixed across a sweep; **α=0 = θ_pre** (pretrained) →
**α=1 = θ_ft** (fully fine-tuned). Each accent's reference goes in the config's `references`
block. The reference *kind* is a **deliberately-varied condition** run in two passes (sibling
trees `results/<accent>/{l1,native}/`; via the Eddie wrapper set `REF_KIND=l1|native`):

- **L1 reference** (paper-faithful): the accent's native-language clip. Because F5 clones accent
  straight from the reference (no language-ID/speaker factorisation), α=0 *already sounds
  accented* — so this pass measures the vector's **marginal** contribution and the α-curve
  **shape** (gain vs leakage), not whether α=0 is accented.
- **Neutral native-English reference** (decoupling control): a clip whose accent is *not* the
  target, so α=0 is neutral English and any target-accent signal that emerges as α climbs is
  **the vector's alone**. This isolates the fine-tuning from reference cloning — the
  discriminating test of RQ1 (does the vector do anything the base model can't?).

**Data.**
- *Train (A0):* per accent, ~100 h of **native-language (L1)** speech — one dataset or several
  combined into a single `audio_file\|text` CSV (use **absolute** audio paths so one
  `--audio-root` covers all sources), then `data_preprocess prepare`. Non-Latin L1 transcripts
  (Hindi/Arabic/Korean) must be romanised (or the vocab extended) first — F5's base vocab won't
  tokenise them. 100 h is ample for a rank-16 LoRA vector; the constraint is GPU-hours.
- *Test:* a **bilingual** corpus (each speaker recorded in their L1 **and** in English) is ideal —
  the L1 utterances are the cloning **references** (A1) and the natural **English** recordings are
  the target-accent clips for `cs_accent` / PPG-KL / F0 (A2), same speaker for both. A
  code-switching corpus works but needs segmenting into clean L1 vs English spans. Keep test
  speakers **disjoint** from the A0 fine-tuning set (else speaker acoustics leak into the scores).
  Optional: set the synth `gen_text` to the speakers' own English sentences for content-matched
  natural-vs-synth pairs.
- *Multiple speakers per accent* (consistency check): give the accent's `references` block one
  entry **per speaker** — `"references": {"indian": {"p1": {...}, "p2": {...}}}` — and the grid
  runs each speaker's sweep into `results/indian/<speaker>/`. Score each speaker with the rq*
  modules (its own L1 reference + natural English), then pool them across speakers with
  `experiments.aggregate` (writes `by_speaker.csv` + `aggregate.csv` = per-α mean ± spread; a
  small spread ⇒ consistent across speakers). No `lora_mapping` needed — single-accent vectors
  default to LoRA idx 0.

## Experiments (CPU / Mac, over the A1 grid)

| ID | RQ | Module | Output | Confirms hypothesis if |
|----|----|--------|--------|------------------------|
| E1.1 | RQ1 | `rq1_reproduction` | `rq1.csv` | accent (`accent_cs`) ↑ monotonic with α (Spearman>0), spk-sim flat/high |
| E1.2 | RQ1 | `rq1_reproduction` (wer col) | `rq1.csv` | WER rises with α faster than paper's XTTS (leakage) |
| E1.3 | RQ1 | `rq1_reproduction --lid` (eng_lid col) | `rq1.csv` | P(English) falls with α — direct language drift, distinct from accent |
| E1.4 | RQ1 | `rq1_reproduction` (leak-onset in footer) | `rq1.csv` | leakage-onset α lower on F5 than XTTS (missing language anchor) |
| E1.5 | RQ1 | `rq1_reproduction` on **both** reference passes (`{l1,native}/`) | `rq1.csv` ×2 | accent ↑ with α under the **neutral** reference ⇒ vector adds accent independent of the reference; flat/leaky ⇒ L1-condition accent was cloning (valid negative result) |
| E2.1 | RQ2 | `rq2_temporal` | `temporal.csv` | `cos(τ_t, τ_final)` converges before magnitude (direction learnable early) |
| E2.2 | RQ2×RQ1 | `checkpoint_grid` → `rq1_reproduction` per step → `rq2_behavioural` | `by_step_summary.csv`, `*_by_step_alpha.csv`, `matched_alpha_trends.csv` | accent (`accent_cs`) saturates with step at low-mid α while `wer` rises and `wer_leak_onset` **falls** with step ⇒ accent learned before language; earlier checkpoint + moderate α is the fluent-accented sweet spot |
| E3.1 | RQ3 | `rq3_decomposition` (seg cols) | `rq3.csv` | PPG-KL-to-natural falls with α |
| E3.2 | RQ3 | `rq3_decomposition` (supra cols) | `rq3.csv` | F0/rhythm move little toward natural |
| E3.3 | RQ3 | `rq3_decomposition` (closure) | `rq3.csv` | seg_closure ≫ supra_closure_mean, widest for distant accent |
| E3.4 | RQ3 | `rq3_layers` | `rq3_layers.csv` | accent energy concentrates in identifiable modules/depth |
| E4.1 | RQ4* | `infer_accent --include-layers/--exclude-layers` (LoRA-native mask) or `extract_vector compose --include` (merged) → `rq3_decomposition` | `rq3.csv` | scaling only accent layers (e.g. `--exclude-layers text_embed input_embed`) keeps English fluent (wer flat) while raising accent; up-weighting prosody layers raises supra_closure |
| E5.1 | RQ5 | `rq5_geometry` | `weight_space_cosine.csv`, `..._mds.csv` | accents cluster by family in MDS |
| E5.2 | RQ5 | `rq5_geometry` (--synth) | `output_space_cosine.csv`, `rsa_mantel.txt` | Mantel r>0, p<0.05 but r<1 (imperfect) |
| E5.3 | RQ5 | `rq5_geometry` (within- vs cross-English) | matrices | corpus contributes measurable distance |

`*` RQ4 is the stretch tier. **E2.1 is Tier-1 only** (optimisation trajectory,
near-free): needs intermediate `model_<step>.pt` checkpoints saved during A0.
The data-efficiency variant (Tier 2/3: separate LoRAs on data fractions with
error bounds) is out of scope — step ≠ data amount.

## Not yet wired (documented integration points)

- **LID probability** (E1.3) — **wired.** `rq1_reproduction --lid` calls
  `evaluation_functions.predict_lid_english`, which runs VoxLingua107
  (`speechbrain/lang-id-voxlingua107-ecapa`) in the isolated `genaid` env via the
  `recipes/CommonAccent/predict_lid.py` wrapper and returns P(English) per clip →
  activates the `eng_lid` column and the LID-based leakage onset. WER still
  provides a parallel (accent-confounded) leakage signal.
- **XTTS token ablation** (RQ1 clean isolation) — out of scope here (needs XTTS
  re-stood-up); the F5-vs-XTTS onset gap confounds backbone with the missing
  language-ID token, so report it as evidence, not proof. See PROPOSAL.md RQ1.
- **Forced-alignment rhythm** (%V, ΔC, nPVI) — `rq3` ships a voicing-based proxy
  from `extract_f0`; swap in MFA vowel/consonant intervals for the rigorous form.

## Typical run order

```bash
# after A0 produces the LoRA snapshots and a grid.json listing them:
python -m accent_vector.experiments.grid --config grid.json --lora             # A1 -> results/<accent>/<speaker>/

# E1 + E3 (core): score each speaker with ITS own L1 reference + natural clips, then pool
for s in results/indian/*/; do sp=$(basename "$s")
  python -m accent_vector.experiments.rq1_reproduction --sweep-dir "$s" \
      --transcripts transcripts/eval_transcripts.txt --ref-wav refs/indian/$sp.wav \
      --accent-ref natural/indian/$sp --lid --out-csv "$s/rq1.csv"
  python -m accent_vector.experiments.rq3_decomposition --sweep-dir "$s" \
      --natural-ref natural/indian/$sp --out-csv "$s/rq3.csv"
done
python -m accent_vector.experiments.aggregate --accent-dir results/indian --csv-name rq1.csv --out-dir results/indian
python -m accent_vector.experiments.aggregate --accent-dir results/indian --csv-name rq3.csv --out-dir results/indian

python -m accent_vector.experiments.rq3_layers --vector vectors/indian.pt \
    --out-csv results/indian/rq3_layers.csv                                            # E3.4 (vector-only)
python -m accent_vector.experiments.rq5_geometry --vector indian=vectors/indian.pt \
    --vector spanish=vectors/spanish.pt --synth indian=results/indian/p1/alpha_1.0 \
    --synth spanish=results/spanish/s1/alpha_1.0 --out-dir results/geometry            # E5
python -m accent_vector.experiments.rq2_temporal --lora \
    --ckpt-dir exps/F5TTS_v1_LoRA_indian/<run>/ckpts/snapshots \
    --out-csv results/indian/temporal.csv                                             # E2.1
```

### Layer-masked accent (E4.1) — "accent without the language"

On the LoRA track the mask is native: `set_lora_alpha` zeroes the un-selected
branches, so `infer_accent` (and `checkpoint_grid`) take `--include-layers` /
`--exclude-layers` (substrings over LoRA submodule names — `attn`, `ff`, `conv`,
`text_embed`, `input_embed`, `lora_proj_out`, or a block index like `.5.`). Excluding
the content/language path is the direct attack on the α=1 gibberish:

```bash
# scale accent-carrying layers only; leave the text/content path at theta_pre
python -m accent_vector.infer_accent --lora \
    --pretrained ckpts/F5TTS_v1_Base/model_1250000.pt --lora-vector vectors/dutch.pt \
    --config <run>/config.yaml --vocab <run>/vocab.txt --alphas 0,0.5,1.0 \
    --exclude-layers text_embed --exclude-layers input_embed \
    --ref-audio refs/native_ga.wav --ref-text "..." \
    --transcripts transcripts/eval_transcripts.txt --out-dir results/dutch/masked
# then score results/dutch/masked with rq1/rq3 as usual and compare to the unmasked sweep
```

### Checkpoint × alpha comparison (E2.2) — accent-vs-language over training

Needs `snapshot_per_updates` snapshots (or full `model_<step>.pt`) from A0. Render
each checkpoint's sweep, score each, then collate at matched α — ideally against the
**neutral** reference so an accent rise is the vector, not cloning:

```bash
# GPU: alpha sweep at several checkpoints -> results/dutch/native/by_step/step_<step>/
python -m accent_vector.experiments.checkpoint_grid \
    --pretrained ckpts/F5TTS_v1_Base/model_1250000.pt \
    --config <run>/config.yaml --vocab <run>/vocab.txt \
    --snap-dir <run>/ckpts/snapshots --steps 5000,15000,25000,45000 \
    --ref-audio refs/native_ga.wav --ref-text-file refs/native_ga.txt \
    --transcripts transcripts/eval_transcripts.txt --alphas 0,0.25,0.5,0.75,1.0 \
    --out-root results/dutch/native/by_step

# CPU: score each checkpoint, then compare matched-alpha across training
for s in results/dutch/native/by_step/step_*/; do
  python -m accent_vector.experiments.rq1_reproduction --sweep-dir "$s" \
    --transcripts transcripts/eval_transcripts.txt --ref-wav refs/native_ga.wav \
    --accent-ref natural/dutch --lid --out-csv "$s/rq1.csv"
done
python -m accent_vector.experiments.rq2_behavioural \
    --by-step-dir results/dutch/native/by_step --out-dir results/dutch/native/trajectory
```

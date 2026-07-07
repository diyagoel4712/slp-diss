# Accent Vector on F5-TTS — adaptation plan & checklist

Goal: replicate the **Accent Vector** method (Lertpetchpun et al., 2026 — task-vector accent
control via LoRA) by adapting **Expressive-Vectors** (github.com/the-bird-F/Expressive-Vectors,
Apache-2.0), but on **F5-TTS** as the backbone instead of the paper's XTTS-v2.

Why F5-TTS: it was the best model in our own benchmark (MCD 7.2, accent-sim 0.73, spk-sim
0.62), we already run it (`SOTA_models_experiments/f5-tts`, weights cached), and Expressive-
Vectors already implements the LoRA-fine-tune → extract-vector → scale/interpolate → infer
pipeline on F5-TTS. The task-vector method is backbone-agnostic: `θ = θ_pre + Σ αᵢ·τᵢ`, where
`τᵢ = θ_LoRA(i)` (the i-th accent's LoRA delta).

---

## The language-conditioning question (verified against installed F5-TTS)

The paper's L2 recipe on XTTS = "fine-tune on the target *language's* native speech while
pinning the **language-ID token to English**", so the LoRA delta is an *accent* shift, not a
*language* switch. **F5-TTS has no language-ID token** — verified: no `lang_id`/`language_id`
in the package; it conditions only on (reference audio, reference text, generation text), with
a `char`/`pinyin` tokenizer. Consequences:

1. **No lang-ID knob to pin** → the anchor that keeps content "English" on XTTS doesn't exist
   on F5. Instead, content language is set purely by the **generation text** you feed at
   inference. So the F5 mapping is: LoRA-fine-tune on target-language audio+transcript (delta
   captures that language's acoustics/prosody); at inference feed **English** text so content
   stays English while the merged delta pushes acoustics toward the accent. Conceptually this
   is *cleaner* on F5 (no competing language token), but it is NOT the paper's exact procedure —
   document the deviation.

2. **Script / vocab coverage is the real gotcha.** F5's base vocab covers Latin (incl.
   accented) + pinyin. For L2 accents whose native script is non-Latin, native transcripts
   won't tokenize (unknown chars → id 0):
   - Vietnamese, Spanish → Latin+diacritics → **OK**.
   - Hindi (Devanagari), Arabic (Arabic), Korean (Hangul) → **NOT covered** → need romanized/
     transliterated transcripts, or a vocab extension, before fine-tuning. **Verify the base
     vocab before assuming.**

3. **Reference-audio confounds the accent source.** F5 clones the reference clip, so at
   inference the reference's own accent competes with the LoRA delta. To isolate "accent from
   the vector" (and make the α-sweep meaningful), hold the reference **fixed and neutral**
   (e.g. one native-English reference) across all α, so the vector is the only thing varying.

---

## Sequencing decision — reproduce British first, then L2

Do the mechanism check on the case with **zero language/script confounds**, then add L2:

- **Phase A (mechanism, easiest): British-accented English.** Within-English (Latin, no
  script issue, no cross-language subtlety) and it is the paper's strongest, cleanest result
  (Table 3: accent prob 23%→57%). Fine-tune LoRA on VCTK *England* speech. If α-scaling raises
  accent-sim while speaker-sim stays high, the port works.
- **Phase B (first L2, Latin script): Vietnamese.** No script issue.
- **Phase C (L2, non-Latin): Indian/Hindi.** Worth the romanization effort because Hindi is the
  *only* one of our four accents that GenAID actually classifies → gives a validated accent-ID
  number, not just embedding cosine. Then Arabic, Korean.

---

## Checklist

### 0. Feasibility gate
- [ ] Secure a **CUDA GPU** (repo pins `torch==2.4.0+cu124`; paper used A40 ×8 GPU-h). CPU
      training is infeasible. Eval stays on the Mac.

### 1. Vendor & isolate (mirror the CosyVoice3/coqui pattern)
- [ ] Clone Expressive-Vectors under `AccentVector/Expressive-Vectors/`, add to `.gitignore`.
- [ ] Init its F5-TTS submodule; create a dedicated CUDA conda env from its README.
- [ ] Record the exact setup steps + any patches in this dir (they won't be version-controlled
      inside the gitignored clone — same lesson as GenAID).

### 2. Reproduce the repo's own example (validate the pipeline before changing anything)
- [ ] Run its E-Vector path end-to-end on *their* data: `finetuning_model.sh` →
      `mining_model.sh` (extract vector) → inference. Confirm it trains + infers on your GPU.
- [ ] Read `scripts/finetuning_model.sh` + `f5_tts_lora/` and answer: which linear layers get
      LoRA, what rank, and **how transcripts/audio are formatted** (expect F5's `audio|text`
      CSV → `prepare_csv_wavs.py` → Arrow).
- [ ] We only need the **single-vector (E-Vector)** path + scaling + linear addition — **skip
      the hierarchical HE-Vector** stage (the Accent Vector paper is single-vector + linear mix).

### 3. Phase A — British-accented English (minimal reproducible claim)
- [ ] Build `metadata.csv` (`audio_file|text`) from VCTK England speakers; filter like the
      paper (DNSMOS>3.4, dur>3s). Run `prepare_csv_wavs.py`.
- [ ] LoRA fine-tune (start from the paper's hyperparams: rank 16, all linear layers, lr 3e-5,
      Adam; steps as compute allows).
- [ ] Extract the accent vector (= LoRA weights) via the mining script.
- [ ] Inference on held-out **English** transcripts, fixed neutral reference, sweep
      **α ∈ {0,0.2,…,1.0}**.
- [ ] **Reproduce the core claim**: accent-sim/accent-ID ↑ monotonically with α, speaker-sim
      stays high. (This is the go/no-go for the whole port.)

### 4. Reuse OUR eval suite (already covers most paper metrics)
Map `SOTA_models_experiments/evaluation_functions.py` → paper §5:
- [ ] WER → our `wer` (Whisper) ✓
- [ ] UTMOS → our `utmos` ✓
- [ ] Speaker sim → our `speaker_similarity` (ECAPA) — paper uses wavlm-base-plus-sv; either is
      fine, just state which.
- [ ] Accent → our `cs_accent` / `aid_acc` (GenAID) stands in for the paper's VoxProfile
      accent-prob + accent-sim. (GenAID only classifies Indian among our accents → embedding
      cosine is the general metric, exactly as in our benchmark.)
- [ ] Point the eval at the new synthesised tree; no metric code changes needed.

### 5. Phases B/C — extend
- [ ] Vietnamese (Latin) LoRA + α-sweep + eval.
- [ ] Resolve non-Latin script (romanize transcripts / extend vocab); Hindi → Arabic → Korean.
- [ ] Linear composition (paper Eq. 5): `τ = Σ αᵢ·τᵢ` for mixed accents; verify both accent
      probs rise.

---

## Open questions to resolve while auditing the repo
- [ ] Does `finetuning_model.sh` fine-tune F5 on *within-language* dialect only (Sichuan/
      Cantonese/Shanghai are Chinese dialects) — i.e. has the repo ever done the **cross-
      language** case our L2 accents need? If not, that recipe is ours to design (Phase B/C).
- [ ] Exact base-vocab contents — does it include Devanagari/Arabic/Hangul? (Determines romanization need.)
- [ ] How the mining script represents the vector (merged full delta vs stored LoRA A/B) —
      affects how we scale and add.

## Task-vector math (reference)
- Extract:  `τ_accent = θ_ft − θ_pre = θ_LoRA`
- Scale:    `θ = θ_pre + α · τ_accent`   (α = accent strength)
- Mix:      `θ = θ_pre + Σᵢ αᵢ · τ_accent(i)`

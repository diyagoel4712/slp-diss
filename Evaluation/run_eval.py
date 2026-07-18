"""Score the L2-ARCTIC accent-TTS grid: headless, re-runnable evaluation entry point.

Builds the (model, accent, speaker, utterance) manifest from eval_config, scores every
clip that has BOTH a synthesised wav and a reference, and writes evaluation_results.csv.
Safe to run repeatedly while synthesis is still in progress: each run rescans the output
trees, so models/cells that have appeared since the last run are picked up, and the CSV is
overwritten with the current full picture.

Each metric subsystem is isolated in try/except so a missing environment (e.g. genaid or
utmosv2 not set up) drops that column rather than aborting the whole pass.

Requires the main evaluation environment (pandas + the native metrics); it shells out to
the isolated UTMOS and GenAID environments for those two subsystems. Point UTMOS_PYTHON /
GENAID_PYTHON at their interpreters to run on another machine. Set EVAL_SKIP_UTMOS=1 to
defer the slow UTMOS pass. See run_pipeline.sh for the end-to-end invocation.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_config as cfg
import evaluation_functions as ef

# utmosv2 lives in its own environment; override with UTMOS_PYTHON on another machine.
UTMOS_PYTHON = os.environ.get("UTMOS_PYTHON", str(cfg.REPO / ".venv" / "bin" / "python"))


def utmos_venv(wav_paths):
    """UTMOS (#1) via the root .venv, with ONE model load for the whole set.

    utmosv2 predicts over a *directory* and keys results by filename. Our clips share names
    across models/speakers (arctic_*.wav), so the previous version grouped by parent dir and
    reloaded the model per directory (~80 loads, ~1.5 h on CPU). Instead we stage every clip
    as a uniquely-named symlink in a single temp dir and predict in one pass. Results are
    mapped back by the unique symlink name OR (if utmosv2 reports the resolved target) by the
    original resolved path -- so it works whichever path form utmosv2 returns.
    """
    import os, shutil, tempfile

    resolved = [str(Path(p).resolve()) for p in wav_paths]
    tmp = tempfile.mkdtemp(prefix="utmos_")
    name2path, resolved_set = {}, set(resolved)
    try:
        for i, rp in enumerate(resolved):
            name = f"u{i:06d}.wav"
            os.symlink(rp, os.path.join(tmp, name))
            name2path[name] = rp
        script = (
            "import sys, json, utmosv2;"
            "m = utmosv2.create_model(pretrained=True, device='cpu');"
            # predict() defaults to device='cuda:0' independently of create_model -> force cpu.
            "print('@@@' + json.dumps(m.predict(input_dir=sys.argv[1], device='cpu')))"
        )
        out = subprocess.run([UTMOS_PYTHON, "-c", script, tmp],
                             capture_output=True, text=True, check=True)
        payload = next(l for l in out.stdout.splitlines() if l.startswith("@@@"))[3:]
        scores = {}
        for rec in json.loads(payload):
            fp = rec["file_path"]
            base = Path(fp).name
            if base in name2path:                       # utmosv2 returned the symlink name
                scores[name2path[base]] = rec["predicted_mos"]
            else:                                        # ... or the resolved target path
                rp = str(Path(fp).resolve())
                if rp in resolved_set:
                    scores[rp] = rec["predicted_mos"]
        return scores
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build_df():
    """Manifest over the full grid x all models, filtered to clips that are evaluable now."""
    # make sure the references we need are extracted from the L2-ARCTIC zips.
    for speakers in cfg.ACCENT_SPEAKERS.values():
        for spk in speakers:
            cfg.ensure_wavs(spk, list(cfg.UTTERANCES))

    rows = []
    for model in cfg.MODELS:
        for cell in cfg.grid():
            rows.append({
                "model": model, **{k: cell[k] for k in ("accent", "speaker", "utt_id", "text")},
                "synth": str(cfg.synth_path(model, cell["accent"], cell["speaker"], cell["utt_id"])),
                "ref":   str(cfg.ref_path(cell["speaker"], cell["utt_id"])),
            })
    manifest = pd.DataFrame(rows)
    manifest["evaluable"] = (manifest["synth"].map(lambda p: Path(p).exists())
                             & manifest["ref"].map(lambda p: Path(p).exists()))
    print("evaluable clips per model:")
    print(manifest.groupby("model")["evaluable"].sum().to_string())
    return manifest[manifest["evaluable"]].reset_index(drop=True).copy()


def per_pair(df, fn, label, second="ref"):
    """Apply a pairwise metric fn(synth, x) over df rows, isolating per-clip failures."""
    vals = []
    for _, r in df.iterrows():
        try:
            vals.append(float(fn(r["synth"], r[second])))
        except Exception as e:
            print(f"  [{label}] {r['model']}/{r['speaker']}/{r['utt_id']}: {e}")
            vals.append(np.nan)
    return vals


def section(name, thunk):
    """Run one metric subsystem; on failure warn and leave its column(s) absent."""
    try:
        thunk()
        print(f"  [{name}] done")
    except Exception as e:
        print(f"  [{name}] SKIPPED ({type(e).__name__}: {e})")


# Order the metric subsystems run in. Each maps to one entry in metric_columns(); the
# name is also the try/except section label. Grouped by the environment they need:
# UTMOS (subprocess), native (this env), then the GenAID subprocess metrics.
METRIC_ORDER = ["utmos", "f0_rmse", "mcd", "wer", "ppg_kl", "aid", "cs_accent", "speaker_sim"]


def metric_columns(df, metric):
    """Compute one metric over the manifest rows, returning {column_name: values}.

    ``df`` must carry the columns each metric consumes: ``synth`` (all), ``ref``
    (paired metrics), ``text`` (WER), and ``accent`` (accent-ID correctness). This is
    the single definition of every metric column, shared by the full run (main) and by
    backfill_metrics.py so a backfilled column matches a full run exactly.
    """
    synth = df["synth"].tolist()
    if metric == "utmos":
        mos = utmos_venv(synth)
        return {"utmos": [mos.get(str(Path(p).resolve()), np.nan) for p in synth]}
    if metric == "f0_rmse":
        return {"f0_rmse": per_pair(df, ef.f0_rmse, "f0_rmse")}
    if metric == "mcd":
        return {"mcd": per_pair(df, ef.mcd, "mcd")}
    if metric == "wer":
        return {"wer": per_pair(df, ef.wer, "wer", second="text")}
    if metric == "ppg_kl":
        return {"ppg_kl": per_pair(df, ef.ppg_kl, "ppg_kl")}
    if metric == "aid":
        preds = ef.predict_accent_genaid(synth)
        by_wav = {p["wav"]: p["pred_accent"] for p in preds}

        def lookup(p):
            raw = by_wav.get(p) or by_wav.get(str(Path(p).resolve()))
            return ef.GENAID_TO_VCTK.get(raw, raw)

        pred = df["synth"].map(lookup)
        # aid_correct is only meaningful for accents GenAID actually labels (Indian).
        return {"aid_pred": pred.tolist(), "aid_correct": (pred == df["accent"]).tolist()}
    if metric == "cs_accent":
        _, sims = ef.cs_accent(synth, df["ref"].tolist(), return_per_pair=True)
        return {"cs_accent": sims}
    if metric == "speaker_sim":
        _, sims = ef.speaker_similarity(synth, df["ref"].tolist(), return_per_pair=True)
        return {"speaker_sim": sims}
    raise ValueError(f"unknown metric {metric!r} (known: {METRIC_ORDER})")


def apply_metric(df, metric):
    """Compute ``metric`` and assign its column(s) onto ``df`` in place."""
    for col, vals in metric_columns(df, metric).items():
        df[col] = vals


def main():
    df = build_df()
    if df.empty:
        print("\nno evaluable clips yet -- nothing to score.")
        return

    for metric in METRIC_ORDER:
        # UTMOS is the slow long pole; EVAL_SKIP_UTMOS=1 defers it (backfill later).
        if metric == "utmos" and os.environ.get("EVAL_SKIP_UTMOS") == "1":
            print("  [utmos] SKIPPED (EVAL_SKIP_UTMOS=1)")
            continue
        section(metric, lambda m=metric: apply_metric(df, m))

    # --- persist + summarise ---
    metric_cols = ["utmos", "f0_rmse", "mcd", "wer", "aid_correct", "cs_accent", "ppg_kl", "speaker_sim"]
    present = [c for c in metric_cols if c in df.columns]
    out_csv = cfg.SOTA / "evaluation_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nwrote {len(df)} rows x {len(present)} metrics -> {out_csv}")

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\nper-model means:")
    print(df.groupby("model")[present].mean(numeric_only=True).to_string())
    print("\nper-accent means (across models):")
    print(df.groupby("accent")[present].mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()

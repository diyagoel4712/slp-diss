"""Unfinished analyses — written, never run to completion, kept for the write-up.

This package is NOT part of the working pipeline. Everything the dissertation
actually depends on lives one level up in ``accent_vector/``:

    data_preprocess -> (finetune on Eddie) -> extract_vector -> infer_accent
                    -> score_sweep / score_prosody -> ../notebooks/dissertation_figures.ipynb

The modules below were written against the experiment matrix
but produced no output in this checkout: none of ``temporal.csv``,
``by_step_summary.csv``, ``weight_space_cosine.csv``, ``rq3_layers.csv``,
``gram.csv`` or ``aggregate.csv`` exists anywhere under AccentVector/. They are
kept because the questions are still open, not because they are wired up.

    evaluate.py         whole-sweep scorer writing ONE metrics.csv -- superseded by
                        score_sweep / score_prosody, which write the rq1.csv /
                        rq3.csv the eval array and the notebook actually use
    aggregate.py        pool per-speaker metric CSVs into a cross-speaker summary
    rq2_behavioural.py  output metrics at matched alpha across checkpoints. Its
                        inputs (per-step rq1.csv) now EXIST, so this is the one
                        module here that could run today -- the notebook's Ch.4
                        trajectory section answers the same question by a
                        different route (local slope + settling step).
    rq2_temporal.py     weight-space trajectory: ||tau_t||, direction convergence
    viz_temporal.py     dashboard video of that trajectory (needs rq2_temporal)
    rq3_layers.py       which modules/depths carry the accent shift
    rq5_geometry.py     weight-space vs output-space accent similarity, RSA/Mantel
    rq5_gram.py         accent-vector Gram matrix: cosines, eigenspectrum, LOO
    rq5_walkthrough.py  narrated tutorial over rq5_gram's objects

Note the RQ numbers in these filenames predate the current chapter structure:
what they call RQ2 the dissertation now calls RQ1b, and no RQ5 section exists.
Rename them if any of them graduates into the pipeline.

Removed rather than kept: ``grid.py`` and ``checkpoint_grid.py``, which built the
synthesis grid on a GPU. The Eddie shell scripts do that job now
(``submit_*_ckpt_grid.sh`` -> ``eddie_infer_array.sh`` -> ``infer_sweep.sh`` ->
``accent_vector.infer_accent``), and no ``grid.json`` was ever written. They are
in git history if the standalone drivers are ever wanted back.

Plotting lives in ``../../../notebooks/dissertation_figures.ipynb``.
"""

"""Accent Vector on F5-TTS -- unified entry point.

Thin dispatcher over the pipeline stages. Each stage is also runnable on its own
as ``python -m accent_vector.<module>``; this just gives them one front door.

    python main.py data     build-vctk --vctk-root ... --out-csv ...
    python main.py data     prepare    --metadata ... --audio-root ... --out-dir ...
    python main.py vector   extract    --pretrained ... --finetuned ... --out ...
    python main.py vector   compose    --pretrained ... --vector ... --alpha ... --out ...
    python main.py infer    --pretrained ... --vector ... --alphas ... --ref-audio ... ...
    python main.py evaluate --sweep-dir ... --transcripts ... --out-csv ...
    python main.py score-sweep   --sweep-dir ... --out-csv .../rq1.csv
    python main.py score-prosody --sweep-dir ... --natural-ref ... --out-csv .../rq3.csv

``score-sweep`` and ``score-prosody`` write the rq1.csv / rq3.csv that
``notebooks/dissertation_figures.ipynb`` turns into every figure; on Eddie they
are driven by scripts/eddie_eval_array.sh rather than through this dispatcher.

Fine-tuning itself is not wrapped here -- it runs through F5-TTS's own
``finetune_cli.py`` (see scripts/finetune.sh), exactly as Expressive-Vectors does.

The package lives under ``src/`` (src layout); this dispatcher adds it to
sys.path itself, so ``python main.py ...`` works from the AccentVector dir with
no PYTHONPATH set. To invoke a module directly instead, use
``PYTHONPATH=src python -m accent_vector.<module>``.

Requires the F5-TTS source on PYTHONPATH and its conda env active; see README.md.
"""

import runpy
import sys
from pathlib import Path

# src layout: the package lives in src/, which is not on sys.path when this file
# is run directly (python main.py ...). Nothing here is pip-installed, so put it
# there ourselves -- the Eddie wrappers export PYTHONPATH=$ACCENT_DIR/src instead.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

STAGES = {
    "data": "accent_vector.data_preprocess",
    "vector": "accent_vector.extract_vector",
    "infer": "accent_vector.infer_accent",
    "evaluate": "accent_vector.evaluate",
    "score-sweep": "accent_vector.score_sweep",
    "score-prosody": "accent_vector.score_prosody",
}


def usage():
    print(__doc__)
    print("stages:", ", ".join(STAGES))


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        usage()
        raise SystemExit(0)
    stage = sys.argv[1]
    if stage not in STAGES:
        print(f"unknown stage: {stage}\n")
        usage()
        raise SystemExit(2)
    # hand the remaining args to the target module as if it were called directly
    sys.argv = [STAGES[stage]] + sys.argv[2:]
    runpy.run_module(STAGES[stage], run_name="__main__")


if __name__ == "__main__":
    main()

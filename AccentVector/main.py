"""Accent Vector on F5-TTS -- unified entry point.

Thin dispatcher over the pipeline stages. Each stage is also runnable on its own
as ``python -m accent_vector.<module>``; this just gives them one front door.

    python main.py data     build-vctk --vctk-root ... --out-csv ...
    python main.py data     prepare    --metadata ... --audio-root ... --out-dir ...
    python main.py vector   extract    --pretrained ... --finetuned ... --out ...
    python main.py vector   compose    --pretrained ... --vector ... --alpha ... --out ...
    python main.py infer    --pretrained ... --vector ... --alphas ... --ref-audio ... ...
    python main.py evaluate --sweep-dir ... --transcripts ... --out-csv ...

Fine-tuning itself is not wrapped here -- it runs through F5-TTS's own
``finetune_cli.py`` (see scripts/finetune.sh), exactly as Expressive-Vectors does.

Requires the F5-TTS source on PYTHONPATH and its conda env active; see README.md.
"""

import runpy
import sys

STAGES = {
    "data": "accent_vector.data_preprocess",
    "vector": "accent_vector.extract_vector",
    "infer": "accent_vector.infer_accent",
    "evaluate": "accent_vector.evaluate",
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

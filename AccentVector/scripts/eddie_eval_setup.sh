#!/bin/bash
# ONE-TIME Eddie setup for running the CPU eval suite on the cluster (so results
# never leave Eddie). RUN THIS ON AN INTERNET-CAPABLE NODE -- a login node or, better,
# an interactive session (`qlogin -l h_vmem=16G`). Compute nodes have NO internet, so
# every model is downloaded + cached here, then jobs run offline (HF_HUB_OFFLINE=1).
#
# Builds three CPU conda envs and stages all model weights:
#   slp-eval  (py3.11)  WER/PPG-KL/F0/MCD           -> requirements-eval.txt
#   genaid    (py3.10)  cs_accent / speaker-sim / LID (SpeechBrain fork, editable)
#   slp-utmos (py3.11)  UTMOS naturalness            -> utmosv2
#
#   bash scripts/eddie_eval_setup.sh            # build everything
#   STEP=envs   bash scripts/eddie_eval_setup.sh   # just the conda envs
#   STEP=genaid bash scripts/eddie_eval_setup.sh   # just clone+patch+checkpoint GenAID
#   STEP=warm   bash scripts/eddie_eval_setup.sh   # download/cache all models (see note)
set -uo pipefail

ACCENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ACCENT_DIR/.." && pwd)"
EVAL_DIR="$REPO/Evaluation"
GENAID_ROOT="$EVAL_DIR/GenAID"
GENAID_CA="$GENAID_ROOT/recipes/CommonAccent"

# --- names + shared cache location (override to taste) ---
EVAL_ENV=${EVAL_ENV:-slp-eval}
GENAID_ENV=${GENAID_ENV:-genaid}
UTMOS_ENV=${UTMOS_ENV:-slp-utmos}
# HF_HOME on scratch (home has a tight quota; the wav2vec2 phoneme model alone is ~1.2G).
export HF_HOME=${HF_HOME:-/exports/eddie/scratch/$USER/hfcache}
STEP=${STEP:-all}

. /etc/profile.d/modules.sh
module load anaconda
mkdir -p "$HF_HOME" "$ACCENT_DIR/logs"
echo "repo=$REPO  HF_HOME=$HF_HOME  envs={$EVAL_ENV,$GENAID_ENV,$UTMOS_ENV}"

build_envs() {
  echo "== [1/3] slp-eval (WER/PPG/F0/MCD) =="
  conda env list | grep -qE "envs/${EVAL_ENV}$" || conda create -y -n "$EVAL_ENV" python=3.11
  # On Linux x86_64 pyworld ships wheels, so pip alone works (unlike the macOS note in
  # requirements-eval.txt); fall back to conda-forge if the wheel is unavailable.
  conda run -n "$EVAL_ENV" pip install -r "$EVAL_DIR/requirements-eval.txt" \
    || { echo "pip failed; retrying pyworld from conda-forge"; \
         conda install -y -n "$EVAL_ENV" -c conda-forge pyworld=0.3.5; \
         conda run -n "$EVAL_ENV" pip install -r "$EVAL_DIR/requirements-eval.txt"; }

  echo "== [2/3] slp-utmos (UTMOS) =="
  conda env list | grep -qE "envs/${UTMOS_ENV}$" || conda create -y -n "$UTMOS_ENV" python=3.11
  # utmosv2 is NOT on PyPI -- install from git, pinned to the commit locked in ../pyproject.toml
  # + uv.lock (bump both together if you change it).
  conda run -n "$UTMOS_ENV" pip install \
    "utmosv2 @ git+https://github.com/sarulab-speech/UTMOSv2.git@cc2700db57bb83ee13dc31ebe1b868c254e15d09"

  echo "== [3/3] genaid (accent-embed / speaker-sim / LID) =="
  conda env list | grep -qE "envs/${GENAID_ENV}$" || conda create -y -n "$GENAID_ENV" python=3.10
  conda run -n "$GENAID_ENV" pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  conda run -n "$GENAID_ENV" pip install -r "$EVAL_DIR/requirements-genaid.txt"
  # editable speechbrain install happens in setup_genaid (needs the clone present first)
}

setup_genaid() {
  echo "== GenAID clone + patches + wrappers + checkpoint =="
  if [ ! -d "$GENAID_ROOT/.git" ] && [ ! -d "$GENAID_ROOT/speechbrain" ]; then
    git clone https://github.com/jzmzhong/GenAID.git "$GENAID_ROOT"
  else
    echo "  clone already present at $GENAID_ROOT"
  fi

  # --- the 3 fork patches for modern huggingface_hub / torchaudio (see Evaluation/README.md) ---
  fetch="$GENAID_ROOT/speechbrain/pretrained/fetching.py"
  iface="$GENAID_ROOT/speechbrain/pretrained/interfaces.py"
  # #1 hf_hub_download arg was renamed use_auth_token -> token
  [ -f "$fetch" ] && sed -i 's/use_auth_token=use_auth_token/token=use_auth_token/g' "$fetch"
  # #2 from_hparams optional-pymodule fetch: modern hub raises EntryNotFoundError, not ValueError
  if [ -f "$iface" ]; then
    n=$(grep -c 'except ValueError:' "$iface" || true)
    [ "$n" -gt 1 ] && echo "  WARN: $n 'except ValueError:' in interfaces.py; broadening ALL to Exception"
    sed -i 's/except ValueError:/except Exception:/g' "$iface"
  fi
  # #3 (librosa load + classify_batch) already lives in the wrappers -> copied below.

  echo "  copying tracked wrappers into the clone"
  cp "$EVAL_DIR"/genaid_wrappers/predict_*.py "$GENAID_CA"/

  echo "  editable speechbrain install"
  conda run -n "$GENAID_ENV" pip install --editable "$GENAID_ROOT"

  # --- GenAID checkpoint (~1.1G, Google Drive) into recipes/CommonAccent/GenAID_v6/ ---
  if [ ! -f "$GENAID_CA/GenAID_v6/save/accent_encoder.txt" ]; then
    echo "  downloading GenAID checkpoint (gdown)"
    conda run -n "$GENAID_ENV" pip install gdown
    ( cd "$GENAID_CA" && \
      conda run -n "$GENAID_ENV" gdown "https://drive.google.com/uc?id=1slGrpZSu5g-nF7R-QMCmtGcjN3kw7lQj" -O GenAID_ckpt.zip && \
      unzip -o GenAID_ckpt.zip && rm -f GenAID_ckpt.zip )
  else
    echo "  checkpoint already present"
  fi
}

warm_caches() {
  # Download+cache every model by scoring ONE real sweep with internet ON. Reuses the
  # array job's own logic (so whatever it needs, gets cached), then compute nodes run
  # the same script with HF_HUB_OFFLINE=1. Pick any existing sweep dir.
  local sweep
  sweep=$(ls -d "$ACCENT_DIR"/results/dutch/native/f/step_* 2>/dev/null | tail -1)
  [ -n "$sweep" ] || { echo "no dutch sweep found to warm on; set it manually" >&2; return 1; }
  echo "== warming model caches on: $sweep =="
  # build a 1-row manifest for that sweep
  local m="$ACCENT_DIR/logs/warm.tsv"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' dutch f native "$sweep" \
    "$ACCENT_DIR/transcripts/dutch/dutch_f_eval.txt" \
    "$ACCENT_DIR/prompts/GAE/gae_f.wav" \
    "$ACCENT_DIR/ground_truth_refs/dutch/female" > "$m"
  SGE_O_WORKDIR="$ACCENT_DIR" SGE_TASK_ID=1 HF_HUB_OFFLINE=0 \
    bash "$ACCENT_DIR/scripts/eddie_eval_array.sh" "$m"
  echo "caches warmed. Compute-node jobs can now run offline."
}

case "$STEP" in
  all)    build_envs; setup_genaid;
          echo; echo "envs + GenAID ready. Next (still on this internet node): STEP=warm bash scripts/eddie_eval_setup.sh";;
  envs)   build_envs;;
  genaid) setup_genaid;;
  warm)   warm_caches;;
  *) echo "STEP must be all|envs|genaid|warm (got '$STEP')" >&2; exit 1;;
esac

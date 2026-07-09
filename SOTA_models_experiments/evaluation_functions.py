# define evaluation metrics -- to be called in an .ipynb file

# 1. UTMOS
def utmos(input_dir):
    """
    automated mean opinion score (MOS) (Baba et al., 2024)
    IN: str: path to folder containing wav files
    OUT: lst: list of dicts with 'file_path' and 'predicted_mos' keys
    """
    import utmosv2
    model = utmosv2.create_model(pretrained=True)
    mos = model.predict(input_dir=input_dir)
    return mos

# -----------------------------------------------------------------------------------------------------------------------
# 2. F0 RMSE
def f0_rmse(synthesised_audio_file, ground_truth_audio_file, **kwargs):
    """
    Root mean squared error (RMSE) in the fundamental frequency (F0) of synthesised speech and natural speech from corpus.
    Helps detect errors in prosody prediction.

    F0 is warped to the mel scale (librosa.hz_to_mel, HTK formula) before the error is taken, so the RMSE reflects perceptual 
    pitch distance rather than raw Hz. Only frames voiced in BOTH signals contribute.

    IN: str: path to synthesised audio file,
        str: path to ground truth audio file
        **kwargs: passed through to extract_f0 (sr, fmin, fmax, hop_length, ...)
    OUT: float: mel-scale F0 RMSE (np.nan if no commonly-voiced frames)
    """
    import numpy as np
    import librosa

    synth_f0 = extract_f0(synthesised_audio_file, **kwargs)
    natural_f0 = extract_f0(ground_truth_audio_file, **kwargs)

    # align lengths: same utterance, slightly different durations -> truncate to minimum.
    # NB: look into DTW-alignment if durations are drastically different (which they shouldn't be! --
    # if your synth and natural speech durations are super different, you have bigger problems to worry about.)
    n = min(len(synth_f0), len(natural_f0))
    synth_f0, natural_f0 = synth_f0[:n], natural_f0[:n]

    # F0 is only defined where there's pitch: compare frames voiced in both.
    voiced = ~np.isnan(synth_f0) & ~np.isnan(natural_f0)
    if not np.any(voiced):
        return np.nan

    # warping to mel scale (more perceptual than simple log scaling, better handling of 0s)
    synth_mel = librosa.hz_to_mel(synth_f0[voiced], htk=True)
    natural_mel = librosa.hz_to_mel(natural_f0[voiced], htk=True)

    return float(np.sqrt(np.mean((natural_mel - synth_mel) ** 2)))


def extract_f0(audio_file, sr=16000, fmin=65.0, fmax=400.0,
               frame_length=1024, hop_length=256):
    """
    extracts f0 trajectory from audio file using the pYIN algorithm.
    fmin/fmax bound the search to a plausible speech pitch range; consider widening fmax
    (e.g. 600) for higher-pitched voices.

    IN: str: path to audio file
    OUT: arr: F0 in Hz per frame; unvoiced frames are np.nan
    """
    import librosa

    y, sr = librosa.load(audio_file, sr=sr)
    f0, _voiced_flag, _voiced_prob = librosa.pyin(
        y, sr=sr, fmin=fmin, fmax=fmax,
        frame_length=frame_length, hop_length=hop_length,
    )
    # pyin returns np.nan on unvoiced frames, which f0_rmse masks out.
    return f0

# -----------------------------------------------------------------------------------------------------------------------
# 3. MCD
def mcd(synthesised_audio_file, ground_truth_audio_file,
        sr=16000, n_mfcc=25, hop_length=256):
    """
    mel cepstral distortion (MCD) between synthesised and natural speech.
    measures spectral-envelope distance; lower is closer to the reference.

    mel-cepstral coefficients are approximated by MFCCs (c0/energy dropped so the
    metric reflects spectral shape, not loudness), then the two sequences are
    DTW-aligned to handle differing durations before the frame-wise distortion is averaged. 
    Uses the Kubichek formulation: (10/ln10) * sqrt(2 * sum_d (c_d - c_d')^2).

    IN: str: path to synthesised audio file,
        str: path to ground truth audio file
        int: sr, n_mfcc (mel-cepstral order + 1), hop_length
    OUT: float: mean MCD in dB
    """
    import numpy as np
    import librosa

    def get_mfccs(audio_file):
        """
        Generates MFCCs from an audio file.

        IN: str: path to audio file
        OUT: arr: array of MFCCs
        """
        y, _ = librosa.load(audio_file, sr=sr)
        m = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
        return m[1:]  # drop c0 (energy); shape (n_mfcc-1, T)

    synth = get_mfccs(synthesised_audio_file)
    natural = get_mfccs(ground_truth_audio_file)

    # DTW-align the two cepstral sequences; wp is the warping path of frame-index pairs.
    _, wp = librosa.sequence.dtw(X=synth, Y=natural, metric="euclidean")
    diff = synth[:, wp[:, 0]] - natural[:, wp[:, 1]]

    K = (10.0 / np.log(10)) * np.sqrt(2.0)
    per_frame = K * np.sqrt(np.sum(diff ** 2, axis=0))
    return float(np.mean(per_frame))

# -----------------------------------------------------------------------------------------------------------------------
# 4. WER
def wer(wav_file, reference):
    """
    word error rate (WER) measure for intelligibility

    I don't think this metric is particularly relevant: we don't expect architecture changes to impact intelligibility, 
    so any significant differences in WER may instead reveal bias in the chosen ASR system.

    IN: str: path to synthesised audio_file
        str: reference trasncription
    OUT: float: wer
    """
    from jiwer import wer as jiwer_wer
    from whisper.normalizers import EnglishTextNormalizer

    hypothesis = get_hypothesis(wav_file=wav_file)

    # lowercases, strips punctuation, canonicalises numbers/contractions/spelling variants.
    normalize = EnglishTextNormalizer()
    error_rate = jiwer_wer(normalize(reference), normalize(hypothesis))

    return error_rate

# cache the Whisper model so a whole test set doesn't reload it per file
_WHISPER_MODEL = None

def get_hypothesis(wav_file, model_size="base.en"):
    """
    automatic transcription derived using Whisper ASR

    IN: str: path to synthesised audio_file
        str: Whisper model size (e.g. "base.en", "small.en", "medium.en")
    OUT: str: hypothesis transcription
    """
    global _WHISPER_MODEL
    import whisper

    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = whisper.load_model(model_size)
    return _WHISPER_MODEL.transcribe(wav_file)["text"]

# -----------------------------------------------------------------------------------------------------------------------
# 5. AccentID acc.

# GenAID runs in its own isolated env (speechbrain 0.5.x / py3.10);
# (calling as a subprocess so its old deps don't collide with this eval env.)
import os as _os
_GENAID_DIR = _os.path.join(_os.path.dirname(__file__), "GenAID", "recipes", "CommonAccent")
_GENAID_PYTHON = "/Users/diyagoel/miniconda3/envs/genaid/bin/python"

# Match accents between GenAID and VCTK
GENAID_TO_VCTK = {
    "us": "American", "canadian": "Canadian", "australian": "Australian",
    "southasian": "Indian", "english": "English", "southernafrican": "SouthAfrican",
    "irish": "Irish", "scottish": "Scottish", "newzealand": "NewZealand"
}

def predict_accent_genaid(wav_files, device="cpu", with_embeddings=False):
    """
    run GenAID on a list of wavs (in its isolated env) -> list of dicts with
    'wav', 'pred_accent', 'posteriors', and optionally 'embedding'.
    """
    import json, subprocess, tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(wav_files))
        list_path = f.name
    out_path = list_path + ".json"

    cmd = [_GENAID_PYTHON, "predict_GenAID.py",
           "--wav_list", list_path, "--device", device,
           "--wav2vec2_save", "./pretrained_xlsr_large", "--out", out_path]
    if not with_embeddings:
        cmd.append("--no_embedding")

    subprocess.run(cmd, cwd=_GENAID_DIR, check=True)
    with open(out_path) as fh:
        return json.load(fh)

# Secondary accent classifier (SpeechBrain CommonAccent ECAPA) for a sanity check.

# Match accents between CommonAccent and VCTK
COMMONACCENT_TO_VCTK = {
    "us": "American", "england": "English", "australia": "Australian",
    "indian": "Indian", "canada": "Canadian", "scotland": "Scottish",
    "ireland": "Irish", "newzealand": "NewZealand", "wales": "Welsh",
    "african": "SouthAfrican",
}

def predict_accent_commonaccent(wav_files, device="cpu"):
    """
    run the CommonAccent ECAPA classifier (isolated env) -> list of dicts with
    'wav', 'pred_accent', 'posteriors'. Use as the secondary predict_fn in aid_acc.
    """
    import json, subprocess, tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(wav_files))
        list_path = f.name
    out_path = list_path + ".ca.json"

    subprocess.run(
        [_GENAID_PYTHON, "predict_commonaccent.py",
         "--wav_list", list_path, "--device", device, "--out", out_path],
        cwd=_GENAID_DIR, check=True,
    )
    with open(out_path) as fh:
        return json.load(fh)

def aid_acc(synthesised_files, target_accents, predict_fn=predict_accent_genaid,
            label_map=GENAID_TO_VCTK):
    """
    accent-ID accuracy: fraction of synthesised clips whose predicted accent matches
    the intended (self-reported) target accent.

    IN: list[str]: synthesised wav paths
        list[str]: intended VCTK accent label per file (same order)
        callable: wav list -> list of {'pred_accent': ...} dicts (GenAID by default)
        dict|None: maps classifier labels onto the target taxonomy (VCTK)
    OUT: float: accuracy in [0, 1]
    """
    preds = predict_fn(synthesised_files)
    pred_labels = [p["pred_accent"] for p in preds]
    if label_map is not None:
        pred_labels = [label_map.get(lbl, lbl) for lbl in pred_labels]

    correct = sum(p == t for p, t in zip(pred_labels, target_accents))
    return correct / len(target_accents)

# -----------------------------------------------------------------------------------------------------------------------
# 6. CS for accent embeddings
def cs_accent(synthesised_files, ground_truth_files,
              predict_fn=predict_accent_genaid, return_per_pair=False):
    """
    cosine similarity (CS) of GenAID accent embeddings between each synthesised clip
    and its paired ground-truth utterance. Higher = the synthesised accent sits closer
    to the natural reference in GenAID's speaker-agnostic accent space.

    IN: list[str]: synthesised wav paths
        list[str]: ground-truth wav paths (paired, same order)
        callable: wav list -> list of dicts with 'wav' and 'embedding' (GenAID only)
        bool: if True, also return the per-pair similarities
    OUT: float: mean cosine similarity over pairs
         (or (mean, list[float]) if return_per_pair)
    """
    import numpy as np

    # one batched call amortises the model load; map embeddings back by path.
    preds = predict_fn(synthesised_files + ground_truth_files, with_embeddings=True)
    emb = {p["wav"]: np.asarray(p["embedding"], dtype=float) for p in preds}

    sims = []
    for s, g in zip(synthesised_files, ground_truth_files):
        a, b = emb[s], emb[g]
        sims.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))

    mean = float(np.mean(sims))
    return (mean, sims) if return_per_pair else mean

# -----------------------------------------------------------------------------------------------------------------------
# 7. PPG-KL

# cache the phoneme recogniser so a whole test set doesn't reload it per call
_PPG_MODEL = None
_PPG_PROCESSOR = None
_PPG_HUB = "facebook/wav2vec2-lv-60-espeak-cv-ft"  # frame-level IPA phoneme posteriors

def extract_ppg(audio_file, sr=16000):
    """
    extract a phonetic posteriorgram: frame-level posterior distribution over phone
    classes from a (speaker-independent) wav2vec2 phoneme-CTC model.
    IN: str: path to audio file
    OUT: arr: (T, P) — each row a probability distribution over P phone classes
    """
    global _PPG_MODEL, _PPG_PROCESSOR
    import numpy as np
    import librosa
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForCTC

    if _PPG_MODEL is None:
        # feature extractor only (audio normalisation); we read logits directly and
        # never decode to text, so the tokenizer/phonemizer backend isn't needed.
        _PPG_PROCESSOR = AutoFeatureExtractor.from_pretrained(_PPG_HUB)
        _PPG_MODEL = AutoModelForCTC.from_pretrained(_PPG_HUB).eval()

    y, _ = librosa.load(audio_file, sr=sr)
    inputs = _PPG_PROCESSOR(y, sampling_rate=sr, return_tensors="pt")
    with torch.no_grad():
        logits = _PPG_MODEL(**inputs).logits[0]      # (T, P)
    ppg = torch.softmax(logits, dim=-1).cpu().numpy()
    return ppg

def _kl_matrix(p, q):
    """pairwise KL(p_i || q_j) -> (len(p), len(q)). p, q are (n, P) row-stochastic."""
    import numpy as np
    logp, logq = np.log(p), np.log(q)
    neg_entropy = np.sum(p * logp, axis=1)            # (len(p),)
    return neg_entropy[:, None] - p @ logq.T

def ppg_kl(synthesised_file, ground_truth_file, sr=16000, eps=1e-8):
    """
    symmetric KL divergence between the phonetic posteriorgrams of synthesised and
    natural speech, averaged over DTW-aligned frames. Measures segmental pronunciation
    fidelity to the reference; lower is closer.

    NB: the result depends on the PPG model (_PPG_HUB) — like WER, a phone recogniser
    biased toward one accent will read other accents' realisations as divergence.

    IN: str: path to synthesised audio file,
        str: path to ground truth audio file
        float: eps — floor added before normalising, so KL stays finite
    OUT: float: mean symmetric KL over aligned frames
    """
    import numpy as np
    import librosa

    def _ppg(path):
        m = extract_ppg(path, sr=sr) + eps
        return m / m.sum(axis=1, keepdims=True)        # re-normalise after flooring

    p = _ppg(synthesised_file)                         # (T_s, P)
    q = _ppg(ground_truth_file)                        # (T_n, P)

    # symmetric KL as the DTW local cost, so the aligned-path mean IS the metric.
    cost = 0.5 * (_kl_matrix(p, q) + _kl_matrix(q, p).T)
    _, wp = librosa.sequence.dtw(C=cost)
    return float(np.mean(cost[wp[:, 0], wp[:, 1]]))

# -----------------------------------------------------------------------------------------------------------------------
# 8. speaker-similarity
def predict_speaker_embeddings(wav_files, device="cpu"):
    """
    extract ECAPA-TDNN (VoxCeleb) speaker embeddings in the isolated env ->
    list of dicts with 'wav' and 'embedding'.
    """
    import json, subprocess, tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(wav_files))
        list_path = f.name
    out_path = list_path + ".spk.json"

    subprocess.run(
        [_GENAID_PYTHON, "predict_speaker_embeddings.py",
         "--wav_list", list_path, "--device", device, "--out", out_path],
        cwd=_GENAID_DIR, check=True,
    )
    with open(out_path) as fh:
        return json.load(fh)

def speaker_similarity(synthesised_files, ground_truth_files,
                       device="cpu", return_per_pair=False):
    """
    speaker-encoder cosine similarity (SECS): cosine similarity of ECAPA-TDNN speaker
    embeddings between each synthesised clip and its paired reference utterance.
    Higher = synthesised voice closer to the target speaker's identity.

    IN: list[str]: synthesised wav paths
        list[str]: reference wav paths (paired, same order)
        bool: if True, also return the per-pair similarities
    OUT: float: mean cosine similarity over pairs
         (or (mean, list[float]) if return_per_pair)
    """
    import numpy as np

    # one batched call amortises the model load; map embeddings back by path.
    preds = predict_speaker_embeddings(synthesised_files + ground_truth_files, device=device)
    emb = {p["wav"]: np.asarray(p["embedding"], dtype=float) for p in preds}

    sims = []
    for s, g in zip(synthesised_files, ground_truth_files):
        a, b = emb[s], emb[g]
        sims.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))

    mean = float(np.mean(sims))
    return (mean, sims) if return_per_pair else mean

# -----------------------------------------------------------------------------------------------------------------------
"""
utils/audio.py — Audio loading and preprocessing utilities.

Handles:
  - 16kHz resampling
  - Mono conversion
  - Energy-based silence detection (to filter out near-silent phone tokens)
  - Safe segment extraction by time boundaries
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


TARGET_SR = 16_000  # WavLM/wav2vec2 expected sample rate

# WavLM CNN feature extractor downsampling factor
# = product of stride in each CNN layer; gives ~50 fps at 16kHz
WAVLM_FRAME_SHIFT = 320   # samples per frame (~20ms)
WAVLM_FRAME_RATE  = TARGET_SR / WAVLM_FRAME_SHIFT  # ~50 fps


def load_audio(path: str, target_sr: int = TARGET_SR) -> torch.Tensor:
    """
    Load audio file, resample to target_sr, convert to mono.
    Returns (T,) float32 tensor.
    """
    waveform, sr = torchaudio.load(str(path))
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    return waveform.squeeze(0)  # (T,)


def extract_segment(
    waveform: torch.Tensor,
    t_start: float,
    t_end: float,
    sr: int = TARGET_SR,
    pad_ms: float = 5.0,
) -> Optional[torch.Tensor]:
    """
    Extract a time segment from a waveform with optional symmetric padding.

    Args:
        waveform: (T,) full utterance waveform
        t_start, t_end: segment boundaries in seconds
        sr: sample rate
        pad_ms: padding in milliseconds added symmetrically

    Returns:
        (T_seg,) segment waveform, or None if segment is too short
    """
    pad_samples = int(pad_ms / 1000 * sr)
    s_start = max(0, int(t_start * sr) - pad_samples)
    s_end   = min(waveform.shape[0], int(t_end * sr) + pad_samples)
    if s_end - s_start < int(0.015 * sr):  # discard < 15ms
        return None
    return waveform[s_start:s_end]


def segment_rms(waveform: torch.Tensor) -> float:
    """Root mean square energy of a segment."""
    return float(torch.sqrt(torch.mean(waveform ** 2)))


def is_too_quiet(waveform: torch.Tensor, threshold_db: float = -45.0) -> bool:
    """
    Return True if segment is below energy threshold (likely silence or stop closure).
    Useful for filtering out near-silent phone tokens that carry no formant information.
    """
    rms = segment_rms(waveform)
    if rms < 1e-10:
        return True
    db = 20 * np.log10(rms)
    return db < threshold_db


def time_to_frame(t: float) -> int:
    """Convert time in seconds to WavLM frame index."""
    return int(t * WAVLM_FRAME_RATE)


def frame_to_time(f: int) -> float:
    """Convert WavLM frame index to time in seconds."""
    return f / WAVLM_FRAME_RATE


def frames_for_segment(t_start: float, t_end: float) -> Tuple[int, int]:
    """
    Return (start_frame, end_frame) indices into WavLM hidden states
    for a phone segment [t_start, t_end] in seconds.
    """
    return time_to_frame(t_start), time_to_frame(t_end)


def collate_variable_length(
    segments: list[torch.Tensor],
    max_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of 1D waveform tensors to the same length.
    Returns (padded_batch, lengths) where lengths are original lengths.
    """
    lengths = torch.tensor([s.shape[0] for s in segments])
    max_len = int(lengths.max()) if max_length is None else max_length
    padded = torch.zeros(len(segments), max_len)
    for i, seg in enumerate(segments):
        l = min(seg.shape[0], max_len)
        padded[i, :l] = seg[:l]
    return padded, lengths

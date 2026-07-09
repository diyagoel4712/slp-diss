"""
dataset.py — PyTorch Dataset for accent encoder training.

Expected inputs:
  - Raw 16kHz .wav files
  - features.json produced by features.py (utterance-level phonological targets)
"""

import json
import torch
import torchaudio
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


# Feature keys that go to regression heads (continuous)
REGRESSION_KEYS = ["f1_mean_norm", "f2_mean_norm", "f3_mean_norm", "f0_mean", "vot_mean"]

# Feature keys that go to classification heads (binary: is/isn't)
CLASSIFICATION_KEYS = ["rhoticity_rate", "flap_rate", "glottal_rate"]
CLASSIFICATION_THRESHOLD = 0.3  # rate > threshold → positive class


class AccentDataset(Dataset):
    """
    Args:
        audio_dir:     Directory containing utterance .wav files.
        features_path: Path to features.json (from features.py).
        speaker_to_id: Dict mapping speaker_id string → int index.
        max_duration:  Clip audio to this many seconds (avoids OOM).
        augment:       Apply SpecAugment-style time/frequency masking.
    """

    def __init__(
        self,
        audio_dir: str,
        features_path: str,
        speaker_to_id: dict,
        max_duration: float = 10.0,
        sample_rate: int = 16000,
        augment: bool = False,
    ):
        self.audio_dir = Path(audio_dir)
        self.speaker_to_id = speaker_to_id
        self.max_samples = int(max_duration * sample_rate)
        self.sample_rate = sample_rate
        self.augment = augment

        with open(features_path) as f:
            all_features = json.load(f)

        # Filter to utterances with audio files present
        self.items = []
        for feat in all_features:
            uid = feat["utterance_id"]
            wav = self.audio_dir / (uid + ".wav")
            if wav.exists() and feat["speaker_id"] in speaker_to_id:
                self.items.append(feat)

        print(f"Dataset: {len(self.items)} utterances, "
              f"{len(set(f['speaker_id'] for f in self.items))} speakers")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        feat = self.items[idx]
        uid = feat["utterance_id"]
        wav_path = self.audio_dir / (uid + ".wav")

        # Load audio
        waveform, sr = torchaudio.load(str(wav_path))
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(0)  # mono

        # Clip
        if waveform.shape[0] > self.max_samples:
            # Random crop during training, centre crop for eval
            if self.augment:
                start = torch.randint(0, waveform.shape[0] - self.max_samples, (1,)).item()
            else:
                start = (waveform.shape[0] - self.max_samples) // 2
            waveform = waveform[start:start + self.max_samples]

        # Build regression target vector (NaN → mask out in loss)
        reg_targets = torch.tensor(
            [self._safe_float(feat.get(k)) for k in REGRESSION_KEYS],
            dtype=torch.float32,
        )

        # Build classification target vector (rate → binary label)
        cls_targets = torch.tensor(
            [int((feat.get(k) or 0.0) > CLASSIFICATION_THRESHOLD)
             for k in CLASSIFICATION_KEYS],
            dtype=torch.long,
        )

        speaker_id = torch.tensor(
            self.speaker_to_id[feat["speaker_id"]], dtype=torch.long
        )

        return {
            "waveform": waveform,
            "reg_targets": reg_targets,    # (len(REGRESSION_KEYS),)
            "cls_targets": cls_targets,    # (len(CLASSIFICATION_KEYS),)
            "speaker_id": speaker_id,
            "utterance_id": uid,
        }

    @staticmethod
    def _safe_float(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return float("nan")
        return float(v)


def collate_fn(batch):
    """Pad waveforms to same length in batch."""
    max_len = max(b["waveform"].shape[0] for b in batch)
    waveforms = []
    attention_masks = []
    for b in batch:
        w = b["waveform"]
        pad = max_len - w.shape[0]
        waveforms.append(torch.nn.functional.pad(w, (0, pad)))
        mask = torch.ones(max_len)
        mask[w.shape[0]:] = 0
        attention_masks.append(mask)

    return {
        "waveform": torch.stack(waveforms),
        "attention_mask": torch.stack(attention_masks),
        "reg_targets": torch.stack([b["reg_targets"] for b in batch]),
        "cls_targets": torch.stack([b["cls_targets"] for b in batch]),
        "speaker_id": torch.stack([b["speaker_id"] for b in batch]),
        "utterance_id": [b["utterance_id"] for b in batch],
    }


def build_speaker_map(features_path: str) -> dict:
    with open(features_path) as f:
        features = json.load(f)
    speakers = sorted(set(d["speaker_id"] for d in features))
    return {s: i for i, s in enumerate(speakers)}

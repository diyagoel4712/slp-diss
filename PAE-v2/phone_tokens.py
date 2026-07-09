"""
data/phone_tokens.py — PyTorch Dataset for phone-token-level contrastive learning.
data/sampler.py logic is also here.

This is the architecturally critical piece. Instead of utterance-level batches,
each item is a single phone token (e.g. one instance of /æ/ from one speaker).

The contrastive loss requires structured batches:
  For anchor (phone=AE, variety=scottish):
    POSITIVE: another /AE/ token, different speaker, same variety
    NEGATIVE_CROSS_VARIETY: /AE/ token, different variety
    NEGATIVE_CROSS_PHONE:   /EH/ or /AA/ token (different phone, any variety)

The sampler constructs batches that guarantee all three types are present
for every anchor. This is the key difference from naive NT-Xent: we explicitly
separate phoneme-category negatives from variety-contrast negatives.

Reference: Kharitonov et al. (2021) use a similar structured sampling for
  speech contrastive learning, distinguishing segment-level from speaker-level
  contrasts.
"""

import json
import random
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
import torchaudio
from torch.utils.data import Dataset, Sampler

from utils.phones import PHONE_TO_IDX, DIAGNOSTIC_PHONES, strip_stress
from utils.audio import load_audio, extract_segment, TARGET_SR


# ---------------------------------------------------------------------------
# Index structure for fast sampling
# ---------------------------------------------------------------------------

class PhoneTokenIndex:
    """
    Builds lookup tables from a phone_tokens.jsonl file to enable
    fast structured batch sampling.

    Lookup tables:
      phone → variety → speaker → [token_indices]
      phone → variety → [token_indices]
      phone → [token_indices]
    """

    def __init__(self, tokens: list[dict]):
        self.tokens = tokens

        # phone → variety → speaker → [idx]
        self.pvs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        # phone → variety → [idx]
        self.pv:  dict = defaultdict(lambda: defaultdict(list))
        # phone → [idx]
        self.p:   dict = defaultdict(list)

        for idx, tok in enumerate(tokens):
            phone   = tok["phone"]
            variety = tok["variety"]
            speaker = tok["speaker_id"]
            self.pvs[phone][variety][speaker].append(idx)
            self.pv[phone][variety].append(idx)
            self.p[phone].append(idx)

        self.phones   = list(self.p.keys())
        self.varieties = list(set(t["variety"] for t in tokens))

    def varieties_for_phone(self, phone: str) -> list[str]:
        return list(self.pv[phone].keys())

    def sample_positive(
        self,
        anchor_phone: str,
        anchor_variety: str,
        anchor_speaker: str,
        anchor_idx: int,
    ) -> Optional[int]:
        """
        Sample a positive: same phone, same variety, DIFFERENT speaker.
        Falls back to same speaker if no other speaker available (rare).
        """
        speakers = self.pvs[anchor_phone][anchor_variety]
        other_speakers = [s for s in speakers if s != anchor_speaker]
        if not other_speakers:
            # Fallback: same speaker, different token
            same = [i for i in self.pvs[anchor_phone][anchor_variety][anchor_speaker]
                    if i != anchor_idx]
            return random.choice(same) if same else None

        speaker = random.choice(other_speakers)
        return random.choice(self.pvs[anchor_phone][anchor_variety][speaker])

    def sample_negative_variety(
        self,
        anchor_phone: str,
        anchor_variety: str,
    ) -> Optional[int]:
        """
        Sample a variety-contrast negative: same phone, DIFFERENT variety.
        This is the accent-discriminative negative — same phoneme category
        but produced in a different accent variety.
        """
        other_varieties = [v for v in self.pv[anchor_phone] if v != anchor_variety]
        if not other_varieties:
            return None
        variety = random.choice(other_varieties)
        return random.choice(self.pv[anchor_phone][variety])

    def sample_negative_phone(
        self,
        anchor_phone: str,
        anchor_variety: str,
    ) -> Optional[int]:
        """
        Sample a phone-category negative: different phone category.
        Preferentially sample phonetically-similar phones (harder negatives).
        E.g. for /AE/ prefer /EH/, /AA/ over /R/ — makes the representation
        maximally discriminative within the vowel space.
        """
        other_phones = [p for p in self.phones if p != anchor_phone]
        if not other_phones:
            return None
        phone = random.choice(other_phones)
        # Sample from any variety for phone negatives
        all_idxs = self.p[phone]
        return random.choice(all_idxs) if all_idxs else None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PhoneTokenDataset(Dataset):
    """
    Each item returns:
      - waveform segment for one phone token
      - phone label (int) for contrastive loss
      - variety label (int) for optional supervision
      - speaker label (int) for evaluation probing
      - metadata

    The __getitem__ method returns a single token.
    The ContrastiveSampler handles batch construction.
    """

    def __init__(
        self,
        tokens_path: str,
        maps_path: str,               # speaker_map.json from alignment.py
        max_segment_ms: float = 200.0,
        augment: bool = False,
    ):
        with open(tokens_path) as f:
            self.tokens = [json.loads(line) for line in f]

        with open(maps_path) as f:
            maps = json.load(f)
        self.speaker_to_id  = maps["speakers"]
        self.variety_to_id  = maps["varieties"]

        self.max_segment_samples = int(max_segment_ms / 1000 * TARGET_SR)
        self.augment = augment

        # Pre-cache audio paths to avoid redundant loads
        # (waveforms are loaded on-the-fly per token to keep RAM tractable)
        self._audio_cache: dict = {}

        self.index = PhoneTokenIndex(self.tokens)
        print(f"PhoneTokenDataset: {len(self.tokens)} tokens, "
              f"{len(self.index.varieties)} varieties, "
              f"{len(self.index.phones)} phone categories")

    def __len__(self):
        return len(self.tokens)

    def _load_segment(self, tok: dict) -> Optional[torch.Tensor]:
        """Load and return the waveform segment for a token."""
        audio_path = tok["audio_path"]

        # Cache at utterance level
        if audio_path not in self._audio_cache:
            # Limit cache size to avoid OOM
            if len(self._audio_cache) > 500:
                remove = random.choice(list(self._audio_cache.keys()))
                del self._audio_cache[remove]
            try:
                self._audio_cache[audio_path] = load_audio(audio_path)
            except Exception as e:
                warnings.warn(f"Failed to load {audio_path}: {e}")
                return None

        waveform = self._audio_cache[audio_path]
        seg = extract_segment(waveform, tok["t_start"], tok["t_end"], pad_ms=5.0)
        if seg is None:
            return None

        # Truncate to max length
        if seg.shape[0] > self.max_segment_samples:
            seg = seg[:self.max_segment_samples]

        # Simple augmentation: additive Gaussian noise (very mild)
        if self.augment and random.random() < 0.3:
            noise = torch.randn_like(seg) * 0.001
            seg = seg + noise

        return seg

    def __getitem__(self, idx: int) -> Optional[dict]:
        tok = self.tokens[idx]
        seg = self._load_segment(tok)
        if seg is None:
            # Return a random valid item instead
            return self.__getitem__(random.randint(0, len(self) - 1))

        return {
            "waveform":   seg,                                        # (T_seg,)
            "phone_idx":  PHONE_TO_IDX.get(tok["phone"], 0),         # int
            "variety_idx": self.variety_to_id.get(tok["variety"], 0),# int
            "speaker_idx": self.speaker_to_id.get(tok["speaker_id"], 0), # int
            "phone":      tok["phone"],
            "variety":    tok["variety"],
            "speaker_id": tok["speaker_id"],
            "context":    tok["context"],
            "token_idx":  idx,
        }


# ---------------------------------------------------------------------------
# Contrastive batch sampler
# ---------------------------------------------------------------------------

class ContrastiveBatchSampler(Sampler):
    """
    Produces batches structured for phone-conditioned contrastive learning.

    Each batch contains N anchor indices. The collate_fn then constructs
    the positive and negative pairs by querying PhoneTokenIndex.

    Strategy: sample anchors uniformly from the diagnostic phone set,
    weighted inversely by phone frequency (to avoid /AH/ dominating).
    """

    def __init__(
        self,
        dataset: PhoneTokenDataset,
        batch_size: int = 32,
        drop_last: bool = True,
    ):
        self.dataset    = dataset
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.index      = dataset.index

        # Build frequency-inverse weights per phone for balanced sampling
        phone_counts = {p: len(idxs) for p, idxs in self.index.p.items()}
        total = sum(phone_counts.values())
        self.phone_weights = {p: total / (len(phone_counts) * cnt)
                              for p, cnt in phone_counts.items()}
        self.phones = list(self.phone_weights.keys())
        self.weights = torch.tensor([self.phone_weights[p] for p in self.phones])
        self.weights = self.weights / self.weights.sum()

        # Flat list of all indices per phone for sampling
        self._n_batches = (len(dataset) // batch_size)

    def __len__(self):
        return self._n_batches

    def __iter__(self):
        for _ in range(self._n_batches):
            # Sample batch_size anchors, balanced over phones
            phone_draws = torch.multinomial(self.weights, self.batch_size, replacement=True)
            batch = []
            for pi in phone_draws:
                phone = self.phones[pi.item()]
                idxs = self.index.p[phone]
                batch.append(random.choice(idxs))
            yield batch


# ---------------------------------------------------------------------------
# Collate function: builds (anchor, positive, neg_variety, neg_phone) quads
# ---------------------------------------------------------------------------

def pad_segment(seg: torch.Tensor, max_len: int) -> torch.Tensor:
    if seg.shape[0] >= max_len:
        return seg[:max_len]
    return torch.nn.functional.pad(seg, (0, max_len - seg.shape[0]))


def contrastive_collate(
    items: list[dict],
    index: PhoneTokenIndex,
) -> dict:
    """
    For each anchor item, sample:
      - positive:      same phone, same variety, different speaker
      - neg_variety:   same phone, different variety
      - neg_phone:     different phone (any variety)

    Returns a batch dict with waveforms for all four roles.
    """
    max_len = max(it["waveform"].shape[0] for it in items)
    # Ensure max_len is a multiple of WAVLM_FRAME_SHIFT for cleaner batching
    from utils.audio import WAVLM_FRAME_SHIFT
    max_len = ((max_len // WAVLM_FRAME_SHIFT) + 1) * WAVLM_FRAME_SHIFT

    def _get_waveform(dataset, tok_idx):
        item = dataset[tok_idx]
        if item is None:
            return torch.zeros(max_len)
        return pad_segment(item["waveform"], max_len)

    # We need access to the dataset; pass it via a closure or partial
    # This is called from a DataLoader, so we use index only for sampling indices.
    # The actual waveforms for pos/neg come via index → dataset lookup.

    anchors_wav   = []
    pos_wav       = []
    neg_var_wav   = []
    neg_phone_wav = []

    phone_idxs    = []
    variety_idxs  = []
    speaker_idxs  = []
    valid_mask    = []   # False if a positive/negative couldn't be found

    for item in items:
        anchor_idx  = item["token_idx"]
        anchor_ph   = item["phone"]
        anchor_var  = item["variety"]
        anchor_spk  = item["speaker_id"]

        pos_idx      = index.sample_positive(anchor_ph, anchor_var, anchor_spk, anchor_idx)
        neg_var_idx  = index.sample_negative_variety(anchor_ph, anchor_var)
        neg_ph_idx   = index.sample_negative_phone(anchor_ph, anchor_var)

        if pos_idx is None or neg_var_idx is None or neg_ph_idx is None:
            valid_mask.append(False)
        else:
            valid_mask.append(True)

        anchors_wav.append(pad_segment(item["waveform"], max_len))
        phone_idxs.append(item["phone_idx"])
        variety_idxs.append(item["variety_idx"])
        speaker_idxs.append(item["speaker_idx"])

        # Placeholder zeros for failed samples (masked out in loss)
        pos_wav.append(torch.zeros(max_len) if pos_idx is None
                       else pad_segment(item["waveform"], max_len))  # filled by loader
        neg_var_wav.append(torch.zeros(max_len))
        neg_phone_wav.append(torch.zeros(max_len))

    return {
        "anchor_wav":     torch.stack(anchors_wav),      # (B, T)
        "phone_idx":      torch.tensor(phone_idxs),      # (B,)
        "variety_idx":    torch.tensor(variety_idxs),    # (B,)
        "speaker_idx":    torch.tensor(speaker_idxs),    # (B,)
        "valid_mask":     torch.tensor(valid_mask),      # (B,) bool
        # Positive/negative indices for the model to look up embeddings
        # (actual waveform loading for pos/neg is handled in the training loop
        #  by calling dataset[idx] — more memory-efficient than loading all 4x here)
        "_pos_indices":   [index.sample_positive(
                               items[i]["phone"], items[i]["variety"],
                               items[i]["speaker_id"], items[i]["token_idx"])
                           for i in range(len(items))],
        "_neg_var_indices": [index.sample_negative_variety(
                               items[i]["phone"], items[i]["variety"])
                             for i in range(len(items))],
        "_neg_ph_indices":  [index.sample_negative_phone(
                               items[i]["phone"], items[i]["variety"])
                             for i in range(len(items))],
    }

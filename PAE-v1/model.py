"""
model.py — Phonological Accent Embedding Encoder

Architecture:
  WavLM/wav2vec2 backbone
    → weighted-sum of transformer layers (learnable)
    → mean-pool over time
    → projection head → L2-normalised accent embedding
    → [train only] phonological feature regression heads
    → [train only] GRL → speaker ID classifier (adversarial)

References:
  - WavLM:       Chen et al. 2022 (IEEE JSTSP)
  - wav2vec 2.0: Baevski et al. 2020 (NeurIPS)
  - GRL:         Ganin et al. 2016 (JMLR)
  - Layer probe: Pasad et al. 2021 (ASRU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, Wav2Vec2Model
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(Function):
    """
    Forward pass: identity.
    Backward pass: multiply gradient by -lambda.
    Ganin et al. (2016) — used to make encoder adversarial to speaker labels.
    """
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_

    def set_lambda(self, lambda_):
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# ---------------------------------------------------------------------------
# Phonological feature head names and sizes
# ---------------------------------------------------------------------------

PHON_FEATURES = {
    # Continuous features (regression)
    "f1_norm":      1,   # Normalised F1 formant (Lobanov or Watt-Fabricius)
    "f2_norm":      1,   # Normalised F2 formant
    "f3_norm":      1,   # F3 (rhoticity marker)
    "vot_ms":       1,   # Voice onset time in ms (stop consonants)
    "f0_mean":      1,   # Mean pitch over vowel nucleus
    # Binary features (classification)
    "is_rhotic":    2,   # Non-rhotic vs rhotic (BCE head with 2 logits)
    "is_flapped":   2,   # /t/ → [ɾ] flapping
    "is_glottalled":2,   # /t/ → [ʔ] glottalisation
}

# Which features are classification vs regression
PHON_CLASSIFICATION = {"is_rhotic", "is_flapped", "is_glottalled"}
PHON_REGRESSION = {k for k in PHON_FEATURES if k not in PHON_CLASSIFICATION}


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AccentEncoder(nn.Module):
    """
    Produces a fixed-size accent embedding from raw audio.

    Args:
        backbone:        "wavlm-base" | "wav2vec2-base"
        embedding_dim:   Size of the output accent embedding (default 256)
        num_speakers:    Number of speakers in training set (for adversarial head)
        freeze_feature_extractor: Freeze CNN feature extractor (recommended)
        use_weighted_layers: Learnable weighted sum over transformer layers
                             (like in SUPERB, Yang et al. 2021)
    """

    def __init__(
        self,
        backbone: str = "wavlm-base",
        embedding_dim: int = 256,
        num_speakers: int = 100,
        freeze_feature_extractor: bool = True,
        use_weighted_layers: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.use_weighted_layers = use_weighted_layers

        # --- Backbone ---
        if backbone == "wavlm-base":
            self.backbone = WavLMModel.from_pretrained("microsoft/wavlm-base")
            hidden_size = 768
            num_layers = 12
        elif backbone == "wav2vec2-base":
            self.backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            hidden_size = 768
            num_layers = 12
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        if freeze_feature_extractor:
            # Freeze CNN feature extractor; fine-tune transformer layers
            self.backbone.feature_extractor._freeze_parameters()

        # Learnable weighted sum over transformer layers (SUPERB-style)
        # Pasad et al. (2021) show that different layers capture different
        # linguistic levels — middle layers are richest for phonetics.
        if use_weighted_layers:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.num_layers = num_layers

        # --- Projection head: hidden → accent embedding ---
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, embedding_dim),
        )

        # --- Phonological feature heads ---
        self.phon_heads = nn.ModuleDict({
            name: nn.Linear(embedding_dim, size)
            for name, size in PHON_FEATURES.items()
        })

        # --- Adversarial speaker head ---
        self.grl = GradientReversalLayer(lambda_=1.0)
        self.speaker_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_speakers),
        )

    def set_grl_lambda(self, lambda_: float):
        """Anneal GRL strength during training (start low, increase)."""
        self.grl.set_lambda(lambda_)

    def encode(self, waveform: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        Produce L2-normalised accent embedding from raw waveform.

        Args:
            waveform: (B, T) float32, 16kHz
            attention_mask: (B, T) optional

        Returns:
            embedding: (B, embedding_dim) L2-normalised
        """
        outputs = self.backbone(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=self.use_weighted_layers,
        )

        if self.use_weighted_layers:
            # Stack all transformer layer outputs: (B, T, H, num_layers)
            hidden_states = torch.stack(outputs.hidden_states[1:], dim=-1)
            weights = F.softmax(self.layer_weights, dim=0)
            # Weighted sum: (B, T, H)
            hidden = (hidden_states * weights).sum(-1)
        else:
            hidden = outputs.last_hidden_state  # (B, T, H)

        # Mean-pool over time (ignoring padding if mask provided)
        if attention_mask is not None:
            # Convert waveform mask → frame mask via backbone's downsampling
            frame_mask = self.backbone._get_feature_vector_attention_mask(
                hidden.shape[1], attention_mask
            )
            frame_mask = frame_mask.unsqueeze(-1).float()
            pooled = (hidden * frame_mask).sum(1) / frame_mask.sum(1).clamp(min=1)
        else:
            pooled = hidden.mean(dim=1)  # (B, H)

        projected = self.projector(pooled)           # (B, embedding_dim)
        embedding = F.normalize(projected, dim=-1)   # L2 normalise
        return embedding

    def forward(self, waveform, attention_mask=None):
        """
        Full forward pass (training mode).

        Returns dict with:
          - "embedding":        (B, D) accent embeddings
          - "phon_preds":       dict of phonological predictions
          - "speaker_logits":   (B, num_speakers) adversarial head output
        """
        embedding = self.encode(waveform, attention_mask)

        # Phonological prediction heads
        phon_preds = {
            name: head(embedding)
            for name, head in self.phon_heads.items()
        }

        # Adversarial speaker head (gradient is reversed through GRL)
        reversed_emb = self.grl(embedding)
        speaker_logits = self.speaker_classifier(reversed_emb)

        return {
            "embedding": embedding,
            "phon_preds": phon_preds,
            "speaker_logits": speaker_logits,
        }

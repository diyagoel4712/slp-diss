"""
models/pacem.py — Full PACEM model.

Combines:
  1. WeightedLayerBackbone: WavLM → frame-level representations
  2. PhoneSegmentPooler:    frame-level → phone-token embeddings (per phone boundary)
  3. Projection head:       phone-token embeddings → contrastive space (training)
  4. PhoneCategoryPooler:   phone tokens → accent embedding (inference)

Training mode:
  Input: single phone token waveform segment
  Output: token embedding in contrastive space

Inference mode:
  Input: full utterance waveform + alignment (phone token boundaries)
  Output: utterance-level accent embedding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

from models.backbone import WeightedLayerBackbone, PhoneSegmentPooler
from models.pooler import PhoneCategoryPooler
from utils.audio import frames_for_segment, WAVLM_FRAME_SHIFT, TARGET_SR


class PACEM(nn.Module):
    """
    Phone-conditioned Accent Contrastive Embedding Model.

    Args:
        backbone_name:   WavLM/wav2vec2 variant
        token_dim:       Phone token embedding dimension (contrastive space)
        embedding_dim:   Final accent embedding dimension
        num_phones:      ARPAbet phone inventory size (for auxiliary phone head)
    """

    def __init__(
        self,
        backbone_name: str = "wavlm-base",
        token_dim: int = 256,
        embedding_dim: int = 256,
        num_phones: int = 39,
        freeze_transformer_layers: int = 0,
    ):
        super().__init__()
        self.token_dim     = token_dim
        self.embedding_dim = embedding_dim

        # 1. Backbone
        self.backbone = WeightedLayerBackbone(
            model_name=backbone_name,
            freeze_feature_extractor=True,
            freeze_transformer_layers=freeze_transformer_layers,
        )
        H = self.backbone.hidden_size

        # 2. Frame → phone-segment pooler
        self.segment_pooler = PhoneSegmentPooler(H, token_dim)

        # 3. Projection head for contrastive training
        # Follows SimCLR convention: a nonlinear head on top of the representation.
        # The contrastive loss is applied to the *projected* space (not token_dim),
        # but the token_dim representations are what the pooler uses at inference.
        self.projector = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        # 4. Auxiliary phone prediction head (multi-task)
        # Predicts which phone category the token is — acts as a regulariser
        # that keeps representations phonetically grounded.
        # Loss weight should be small (0.1–0.3) relative to contrastive loss.
        self.phone_head = nn.Linear(token_dim, num_phones)

        # 5. Utterance-level pooler (inference)
        self.accent_pooler = PhoneCategoryPooler(token_dim, embedding_dim)

    # -----------------------------------------------------------------------
    # Training-time forward: single phone token segment
    # -----------------------------------------------------------------------

    def embed_token(
        self,
        waveform: torch.Tensor,           # (B, T) — one phone token per item
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Embed a batch of phone token waveform segments.

        Returns:
            token_emb:    (B, token_dim) — use for pooler at inference
            projected:    (B, token_dim) — use for contrastive loss
            phone_logits: (B, num_phones) — use for auxiliary phone loss
        """
        hidden, frame_mask = self.backbone(waveform, attention_mask)  # (B, T_f, H)

        # For phone token segments, mean-pool the whole segment
        # (the segment IS the phone — no sub-boundary alignment needed)
        if frame_mask is not None:
            mask = frame_mask.unsqueeze(-1).float()          # (B, T_f, 1)
            token_emb = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            token_emb = hidden.mean(1)                       # (B, H)

        token_emb = self.segment_pooler.proj(token_emb)      # (B, token_dim)
        projected = self.projector(token_emb)                 # (B, token_dim)
        projected = F.normalize(projected, dim=-1)

        phone_logits = self.phone_head(token_emb)             # (B, num_phones)

        return {
            "token_emb":    token_emb,
            "projected":    projected,
            "phone_logits": phone_logits,
        }

    # -----------------------------------------------------------------------
    # Inference-time forward: full utterance → accent embedding
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def embed_utterance(
        self,
        waveform: torch.Tensor,                  # (1, T) full utterance
        phone_tokens: List[dict],                # from alignment.py
    ) -> torch.Tensor:
        """
        Produce a single accent embedding for a full utterance.

        Args:
            waveform:     (1, T) full utterance waveform at 16kHz
            phone_tokens: list of dicts with "phone", "t_start", "t_end"

        Returns:
            accent_emb: (embedding_dim,) L2-normalised accent embedding
        """
        self.eval()
        hidden, _ = self.backbone(waveform)  # (1, T_frames, H)
        hidden = hidden.squeeze(0)           # (T_frames, H)

        token_embeddings = []
        phone_labels     = []

        for tok in phone_tokens:
            f_start, f_end = frames_for_segment(tok["t_start"], tok["t_end"])
            token_emb = self.segment_pooler.pool_segment(hidden, f_start, f_end)
            token_embeddings.append(token_emb)
            phone_labels.append(tok["phone"])

        if not token_embeddings:
            return torch.zeros(self.embedding_dim)

        token_stack = torch.stack(token_embeddings)  # (N, token_dim)
        accent_emb  = self.accent_pooler(token_stack, phone_labels)
        return accent_emb

    def forward(self, waveform, attention_mask=None):
        """Default forward = token embedding (training mode)."""
        return self.embed_token(waveform, attention_mask)

"""
models/backbone.py — WavLM backbone with SUPERB-style weighted layer aggregation.

Key design choices:
  1. Freeze CNN feature extractor (standard practice; it's a spectrogram-like
     transform that doesn't benefit from accent fine-tuning)
  2. Unfreeze all transformer layers (accent-relevant phonetic variation is
     encoded in the transformer representations)
  3. Learnable scalar weights over all 12 layers — the optimal mixture for
     accent discrimination may differ from the layer distributions optimal
     for ASR or speaker ID (Pasad et al. 2021)
  4. Output is frame-level (50fps), NOT pooled — pooling is done externally
     by the phone-token pooler, conditioned on alignment boundaries

Reference: Yang et al. (2021), SUPERB, Interspeech.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import WavLMModel, Wav2Vec2Model
from typing import Optional


class WeightedLayerBackbone(nn.Module):
    """
    WavLM or wav2vec2 backbone that outputs a weighted sum of all transformer
    layer hidden states, at frame resolution (~50fps).

    Returns:
        hidden: (B, T_frames, H) where T_frames ≈ T_samples / 320
    """

    SUPPORTED = {
        "wavlm-base":    ("microsoft/wavlm-base",      768, 12),
        "wavlm-base+":   ("microsoft/wavlm-base-plus",  768, 12),
        "wav2vec2-base": ("facebook/wav2vec2-base",     768, 12),
        "wav2vec2-large":("facebook/wav2vec2-large",   1024, 24),
    }

    def __init__(
        self,
        model_name: str = "wavlm-base",
        freeze_feature_extractor: bool = True,
        freeze_transformer_layers: int = 0,  # freeze first N transformer layers
    ):
        super().__init__()

        if model_name not in self.SUPPORTED:
            raise ValueError(f"Unknown backbone '{model_name}'. "
                             f"Choose from: {list(self.SUPPORTED.keys())}")

        hf_name, self.hidden_size, self.num_layers = self.SUPPORTED[model_name]

        if "wavlm" in model_name:
            self.encoder = WavLMModel.from_pretrained(hf_name)
        else:
            self.encoder = Wav2Vec2Model.from_pretrained(hf_name)

        # Freeze CNN feature extractor (always)
        if freeze_feature_extractor:
            self.encoder.feature_extractor._freeze_parameters()

        # Optionally freeze first N transformer layers
        if freeze_transformer_layers > 0:
            for i, layer in enumerate(self.encoder.encoder.layers):
                if i < freeze_transformer_layers:
                    for param in layer.parameters():
                        param.requires_grad_(False)

        # Learnable scalar weights for each transformer layer
        # Initialised to uniform; softmax normalised during forward
        self.layer_weights = nn.Parameter(
            torch.ones(self.num_layers) / self.num_layers
        )

    def get_attention_mask_for_frames(
        self,
        waveform_mask: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Convert a waveform-level attention mask (B, T_samples) to a
        frame-level mask (B, T_frames) matching WavLM's CNN downsampling.

        This is the correct way to propagate masks through the CNN.
        The approach differs between WavLM and wav2vec2 internally, but
        the public API method handles this uniformly.
        """
        return self.encoder._get_feature_vector_attention_mask(
            num_frames, waveform_mask
        )

    def forward(
        self,
        waveform: torch.Tensor,                # (B, T)
        attention_mask: Optional[torch.Tensor] = None,  # (B, T)
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns:
            hidden:       (B, T_frames, H) weighted sum of layer hidden states
            frame_mask:   (B, T_frames) bool mask, or None
        """
        outputs = self.encoder(
            waveform,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # hidden_states[0] is the CNN output (before transformer)
        # hidden_states[1:] are the 12 transformer layer outputs
        # Shape of each: (B, T_frames, H)
        layer_hiddens = torch.stack(outputs.hidden_states[1:], dim=0)  # (L, B, T, H)

        weights = F.softmax(self.layer_weights, dim=0)  # (L,)
        # Weighted sum: (B, T, H)
        hidden = torch.einsum("l,lbth->bth", weights, layer_hiddens)

        # Propagate mask to frame level
        frame_mask = None
        if attention_mask is not None:
            T_frames = hidden.shape[1]
            frame_mask = self.get_attention_mask_for_frames(attention_mask, T_frames)

        return hidden, frame_mask


class PhoneSegmentPooler(nn.Module):
    """
    Pools frame-level WavLM representations to phone-segment-level embeddings
    using forced alignment boundaries.

    This is NOT simple mean-pooling of the whole utterance. For each phone
    token [t_start, t_end], it extracts the corresponding frames and
    mean-pools only those frames.

    Input:
        hidden:      (B=1, T_frames, H) — typically called per utterance
        frame_start: int — first frame of phone segment
        frame_end:   int — last frame of phone segment (exclusive)

    Output:
        phone_emb: (H,) single phone token embedding
    """

    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.GELU(),
            nn.LayerNorm(output_size),
        )

    def pool_segment(
        self,
        hidden: torch.Tensor,    # (T_frames, H)
        frame_start: int,
        frame_end: int,
    ) -> torch.Tensor:
        """Mean-pool frames within phone boundaries."""
        frame_end = min(frame_end, hidden.shape[0])
        frame_start = max(0, frame_start)
        if frame_end <= frame_start:
            frame_end = frame_start + 1
        segment = hidden[frame_start:frame_end]     # (n_frames, H)
        pooled = segment.mean(0)                    # (H,)
        return self.proj(pooled)                    # (output_size,)

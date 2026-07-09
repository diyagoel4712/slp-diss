"""
models/pooler.py — Phone-category pooler: the inference-time aggregation step.

At training time, the model operates on individual phone tokens.
At inference time, we want a single accent embedding for an entire utterance.

The pooler:
  1. Groups all phone token embeddings by their ARPAbet category
  2. Mean-pools within each diagnostic phone category
     (e.g. all /AE/ tokens → one /AE/ centroid)
  3. Concatenates the per-category centroids
  4. Projects to final accent embedding dimension

Key design choices:
  - Only diagnostic phone categories are included (not all 39 phones)
  - Categories with zero tokens in the utterance get a learned "absent" token
    rather than zeros — important because /ER/ being absent is itself
    informative (it distinguishes non-rhotic from rhotic varieties)
  - The absent token is trained jointly, so the model learns to distinguish
    "truly non-rhotic speaker" from "rhotic speaker with no ER tokens in clip"

Theoretical basis:
  The concatenation of per-category centroids preserves the conditional
  distribution structure F1|phone_category that Labov (1994) identifies
  as the locus of accent variation. The final projection learns which
  combination of cross-category relationships is most discriminative
  for the specific accent contrasts in the training data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from utils.phones import DIAGNOSTIC_GROUPS, NUM_DIAGNOSTIC_GROUPS, DIAGNOSTIC_GROUP_MAP


class PhoneCategoryPooler(nn.Module):
    """
    Aggregates a set of phone token embeddings into a single accent embedding.

    Args:
        token_dim:     Dimensionality of individual phone token embeddings
        embedding_dim: Final accent embedding dimensionality
    """

    def __init__(
        self,
        token_dim: int = 256,
        embedding_dim: int = 256,
    ):
        super().__init__()
        self.token_dim     = token_dim
        self.embedding_dim = embedding_dim
        self.n_groups      = NUM_DIAGNOSTIC_GROUPS

        # One learnable "absent category" embedding per diagnostic group.
        # Trained end-to-end so the model learns the representational cost
        # of absence (e.g. non-rhotic speaker will always use this for /ER/).
        self.absent_tokens = nn.Parameter(
            torch.randn(self.n_groups, token_dim) * 0.02
        )

        # Group names in stable order (same as DIAGNOSTIC_GROUPS list)
        self.group_names: List[str] = [g.name for g in DIAGNOSTIC_GROUPS]
        self.group_phones: Dict[str, frozenset] = {
            g.name: g.phones for g in DIAGNOSTIC_GROUPS
        }

        # Phone → group index mapping
        self.phone_to_group_indices: Dict[str, List[int]] = {}
        for gi, g in enumerate(DIAGNOSTIC_GROUPS):
            for phone in g.phones:
                self.phone_to_group_indices.setdefault(phone, []).append(gi)

        # Final projection: concatenated group embeddings → accent embedding
        concat_dim = self.n_groups * token_dim
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,   # (N_tokens, token_dim)
        phone_labels: List[str],          # length N_tokens, ARPAbet labels
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: (N, D) — all phone token embeddings from one utterance
            phone_labels:     list of N ARPAbet phone labels (no stress digits)

        Returns:
            accent_embedding: (embedding_dim,) L2-normalised
        """
        device = token_embeddings.device

        # For each diagnostic group, collect tokens belonging to that group
        group_embeddings = []

        for gi, g in enumerate(DIAGNOSTIC_GROUPS):
            # Gather token indices whose phone belongs to this group
            member_indices = [
                i for i, ph in enumerate(phone_labels)
                if ph in g.phones
            ]

            if member_indices:
                # Mean-pool all tokens in this group
                member_embs = token_embeddings[member_indices]  # (K, D)
                group_emb = member_embs.mean(0)                 # (D,)
            else:
                # Use learned absent token (non-rhotic speaker has no /ER/ tokens)
                group_emb = self.absent_tokens[gi]

            group_embeddings.append(group_emb)

        # Concatenate all group embeddings: (n_groups * token_dim,)
        concat = torch.cat(group_embeddings, dim=0)

        # Project and L2-normalise
        projected = self.projection(concat)                   # (embedding_dim,)
        return F.normalize(projected, dim=0)                  # L2-normalised

    def forward_batch(
        self,
        token_embeddings_list: List[torch.Tensor],   # list of (N_i, D)
        phone_labels_list: List[List[str]],          # list of N_i labels
    ) -> torch.Tensor:
        """
        Batched version: pool multiple utterances.
        Returns: (B, embedding_dim)
        """
        return torch.stack([
            self.forward(embs, labels)
            for embs, labels in zip(token_embeddings_list, phone_labels_list)
        ])

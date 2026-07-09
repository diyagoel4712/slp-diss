"""
training/losses.py — Phone-conditioned NT-Xent loss with cross-phone repulsion.

Standard NT-Xent (SimCLR, Chen et al. 2020) treats all non-positive pairs
as negatives. For accent learning this is wrong: two /AE/ tokens from
different varieties should be somewhat similar (same phoneme category) but
distinguishable (different variety). Two /AE/ and /AA/ tokens should be
more different (different phoneme category).

The phone-conditioned loss separates these:

  L_total = L_variety_contrast + α * L_phone_separation + β * L_phone_aux

  L_variety_contrast:
    For each anchor (phone p, variety v):
      positive   = same p, same v, different speaker
      negatives  = same p, DIFFERENT variety
    → learns to separate accent varieties within a phone category
    → does NOT penalise cross-phone distances (those are handled separately)

  L_phone_separation:
    For each anchor:
      negatives = DIFFERENT phone category (any variety)
    → ensures the space preserves phonemic distinctions
    → acts as a regulariser preventing variety-collapse across phone categories

  L_phone_aux:
    Cross-entropy on the phone prediction head
    → keeps the token representation phonetically grounded
    → prevents degenerate solutions where all tokens converge

Reference:
  Chen et al. (2020). A simple framework for contrastive learning. ICML.
  Kharitonov et al. (2021). Data augmenting contrastive learning of speech. SLT.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def nt_xent_loss(
    anchors: torch.Tensor,       # (B, D) L2-normalised
    positives: torch.Tensor,     # (B, D) L2-normalised
    temperature: float = 0.07,
    negatives: Optional[torch.Tensor] = None,  # (B, K, D) explicit negatives
    valid_mask: Optional[torch.Tensor] = None, # (B,) bool
) -> torch.Tensor:
    """
    NT-Xent loss with optional explicit negatives.

    If negatives is None: in-batch negatives (standard SimCLR).
    If negatives is given: use provided negatives (ignores other batch items as neg).

    With in-batch negatives, each anchor's positive is in the denominator
    as both a positive (numerator) and a negative for other anchors.
    The standard SimCLR formulation.
    """
    B = anchors.shape[0]
    if valid_mask is None:
        valid_mask = torch.ones(B, dtype=torch.bool, device=anchors.device)

    if negatives is not None:
        # Explicit negatives mode: (B, K, D)
        # sim(anchor, positive): (B,)
        pos_sim = (anchors * positives).sum(-1) / temperature     # (B,)
        # sim(anchor, negatives): (B, K)
        neg_sim = torch.bmm(
            anchors.unsqueeze(1),
            negatives.transpose(1, 2)
        ).squeeze(1) / temperature                                # (B, K)

        # log-softmax over [positive, k negatives]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 1+K)
        labels = torch.zeros(B, dtype=torch.long, device=anchors.device)
        loss = F.cross_entropy(logits, labels, reduction="none")    # (B,)
    else:
        # In-batch negatives (SimCLR style)
        # All 2B embeddings in one pool
        all_embs = torch.cat([anchors, positives], dim=0)           # (2B, D)
        sim = torch.mm(all_embs, all_embs.T) / temperature          # (2B, 2B)

        # Mask self-similarities
        mask = torch.eye(2 * B, dtype=torch.bool, device=anchors.device)
        sim.masked_fill_(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.arange(B, device=anchors.device)
        labels_full = torch.cat([labels + B, labels], dim=0)        # (2B,)

        loss = F.cross_entropy(sim, labels_full, reduction="none")
        # Take the first B (anchor→positive direction only)
        loss = loss[:B]

    # Apply valid mask (drop anchors where pos/neg sampling failed)
    loss = loss[valid_mask]
    return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, requires_grad=True)


class PhoneConditionedContrastiveLoss(nn.Module):
    """
    Full PACEM training loss.

    Args:
        temperature_variety: temperature for variety-contrast loss
        temperature_phone:   temperature for phone-separation loss
        alpha:               weight of phone-separation term
        beta:                weight of auxiliary phone prediction term
    """

    def __init__(
        self,
        temperature_variety: float = 0.07,
        temperature_phone:   float = 0.10,
        alpha:               float = 0.3,
        beta:                float = 0.1,
    ):
        super().__init__()
        self.tau_v  = temperature_variety
        self.tau_p  = temperature_phone
        self.alpha  = alpha
        self.beta   = beta
        self.ce     = nn.CrossEntropyLoss()

    def forward(
        self,
        # Anchor embeddings (projected, L2-normalised)
        anchor_proj:      torch.Tensor,   # (B, D)
        # Positive (same phone, same variety, diff speaker)
        pos_proj:         torch.Tensor,   # (B, D)
        # Variety negatives (same phone, diff variety)
        neg_var_proj:     torch.Tensor,   # (B, D) — one per anchor
        # Phone negatives (diff phone, any variety)
        neg_phone_proj:   torch.Tensor,   # (B, D) — one per anchor
        # Phone category labels (for auxiliary loss)
        phone_idx:        torch.Tensor,   # (B,) long
        phone_logits:     torch.Tensor,   # (B, num_phones)
        # Valid mask (where pos/neg sampling succeeded)
        valid_mask:       torch.Tensor,   # (B,) bool
    ) -> dict:
        """
        Returns dict with total loss and per-component losses for logging.
        """
        # ----------------------------------------------------------------
        # L_variety_contrast
        # Positive: same phone, same variety, diff speaker
        # Negative: same phone, diff variety
        # Temperature is low (0.07) — we want tight variety clusters
        # ----------------------------------------------------------------
        neg_var_expanded = neg_var_proj.unsqueeze(1)   # (B, 1, D)
        l_variety = nt_xent_loss(
            anchor_proj,
            pos_proj,
            temperature=self.tau_v,
            negatives=neg_var_expanded,
            valid_mask=valid_mask,
        )

        # ----------------------------------------------------------------
        # L_phone_separation
        # Negative: different phone category
        # Higher temperature (0.10) — phone boundaries should be clear but
        # not infinitely sharp (phonetic gradience is real)
        # ----------------------------------------------------------------
        neg_phone_expanded = neg_phone_proj.unsqueeze(1)  # (B, 1, D)
        l_phone_sep = nt_xent_loss(
            anchor_proj,
            pos_proj,
            temperature=self.tau_p,
            negatives=neg_phone_expanded,
            valid_mask=valid_mask,
        )

        # ----------------------------------------------------------------
        # L_phone_aux: auxiliary phone category prediction
        # ----------------------------------------------------------------
        if valid_mask.any():
            l_phone_aux = self.ce(
                phone_logits[valid_mask],
                phone_idx[valid_mask],
            )
        else:
            l_phone_aux = torch.tensor(0.0, device=anchor_proj.device, requires_grad=True)

        # ----------------------------------------------------------------
        # Total
        # ----------------------------------------------------------------
        l_total = l_variety + self.alpha * l_phone_sep + self.beta * l_phone_aux

        return {
            "loss":          l_total,
            "l_variety":     l_variety.detach(),
            "l_phone_sep":   l_phone_sep.detach(),
            "l_phone_aux":   l_phone_aux.detach(),
        }

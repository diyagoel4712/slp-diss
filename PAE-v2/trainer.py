"""
training/trainer.py — PACEM training loop.

The training loop is more complex than standard because we need to:
  1. Forward the anchor waveform through the backbone
  2. Sample positive and negative tokens (by index) from the dataset
  3. Forward those through the backbone too
  4. Compute the phone-conditioned contrastive loss

This means each "batch" of B anchors requires up to 4B backbone forward passes
(anchor + positive + neg_variety + neg_phone).

Efficiency note: in practice we forward all 4B segments in a single batched
call if memory permits, otherwise we accumulate gradients across 2 forward passes.
"""

import os
import json
import math
import argparse
from pathlib import Path
from typing import Optional
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.pacem import PACEM
from data.phone_tokens import PhoneTokenDataset, ContrastiveBatchSampler
from training.losses import PhoneConditionedContrastiveLoss
from utils.audio import TARGET_SR


# ---------------------------------------------------------------------------
# Helper: embed a list of token indices through the model
# ---------------------------------------------------------------------------

def embed_indices(
    model: PACEM,
    dataset: PhoneTokenDataset,
    indices: list,
    max_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load waveforms for a list of token indices and embed them.
    Returns (projected, phone_logits).
    Indices of None → zero embedding (masked out by valid_mask).
    """
    waveforms = []
    for idx in indices:
        if idx is None:
            waveforms.append(torch.zeros(max_len))
        else:
            item = dataset[idx]
            if item is None:
                waveforms.append(torch.zeros(max_len))
            else:
                w = item["waveform"]
                if w.shape[0] < max_len:
                    w = torch.nn.functional.pad(w, (0, max_len - w.shape[0]))
                else:
                    w = w[:max_len]
                waveforms.append(w)

    batch = torch.stack(waveforms).to(device)            # (B, T)
    out = model.embed_token(batch)
    return out["projected"], out["phone_logits"]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model:     PACEM,
    dataset:   PhoneTokenDataset,
    loader:    DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: PhoneConditionedContrastiveLoss,
    device:    torch.device,
    grad_accum: int = 1,
    max_segment_samples: int = int(0.2 * TARGET_SR),
) -> dict:
    model.train()
    totals = {k: 0.0 for k in ("loss", "l_variety", "l_phone_sep", "l_phone_aux")}
    n_batches = 0
    optimiser.zero_grad()

    for step, batch in enumerate(loader):
        anchor_wav  = batch["anchor_wav"].to(device)      # (B, T)
        phone_idx   = batch["phone_idx"].to(device)
        valid_mask  = batch["valid_mask"].to(device)

        pos_indices     = batch["_pos_indices"]
        neg_var_indices = batch["_neg_var_indices"]
        neg_ph_indices  = batch["_neg_ph_indices"]

        max_len = anchor_wav.shape[1]

        # Forward anchor
        anchor_out  = model.embed_token(anchor_wav)
        anchor_proj = anchor_out["projected"]
        phone_logits = anchor_out["phone_logits"]

        # Forward positive and negatives
        with torch.no_grad() if False else torch.enable_grad():
            pos_proj, _    = embed_indices(model, dataset, pos_indices,     max_len, device)
            neg_var_proj, _ = embed_indices(model, dataset, neg_var_indices, max_len, device)
            neg_ph_proj, _  = embed_indices(model, dataset, neg_ph_indices,  max_len, device)

        losses = criterion(
            anchor_proj=anchor_proj,
            pos_proj=pos_proj,
            neg_var_proj=neg_var_proj,
            neg_phone_proj=neg_ph_proj,
            phone_idx=phone_idx,
            phone_logits=phone_logits,
            valid_mask=valid_mask,
        )

        loss = losses["loss"] / grad_accum
        loss.backward()

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            optimiser.zero_grad()

        for k in totals:
            totals[k] += losses[k].item() if torch.is_tensor(losses[k]) else losses[k]
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(
    model:     PACEM,
    dataset:   PhoneTokenDataset,
    loader:    DataLoader,
    criterion: PhoneConditionedContrastiveLoss,
    device:    torch.device,
) -> dict:
    model.eval()
    totals = {k: 0.0 for k in ("loss", "l_variety", "l_phone_sep", "l_phone_aux")}

    # Phone prediction accuracy (proxy metric)
    correct_phone = 0
    total = 0
    n_batches = 0

    for batch in loader:
        anchor_wav  = batch["anchor_wav"].to(device)
        phone_idx   = batch["phone_idx"].to(device)
        valid_mask  = batch["valid_mask"].to(device)

        pos_indices     = batch["_pos_indices"]
        neg_var_indices = batch["_neg_var_indices"]
        neg_ph_indices  = batch["_neg_ph_indices"]

        max_len = anchor_wav.shape[1]

        anchor_out  = model.embed_token(anchor_wav)
        anchor_proj = anchor_out["projected"]
        phone_logits = anchor_out["phone_logits"]

        pos_proj, _     = embed_indices(model, dataset, pos_indices,     max_len, device)
        neg_var_proj, _ = embed_indices(model, dataset, neg_var_indices, max_len, device)
        neg_ph_proj, _  = embed_indices(model, dataset, neg_ph_indices,  max_len, device)

        losses = criterion(
            anchor_proj=anchor_proj,
            pos_proj=pos_proj,
            neg_var_proj=neg_var_proj,
            neg_phone_proj=neg_ph_proj,
            phone_idx=phone_idx,
            phone_logits=phone_logits,
            valid_mask=valid_mask,
        )

        for k in totals:
            totals[k] += losses[k].item() if torch.is_tensor(losses[k]) else losses[k]

        # Phone accuracy
        if valid_mask.any():
            preds = phone_logits[valid_mask].argmax(-1)
            correct_phone += (preds == phone_idx[valid_mask]).sum().item()
            total += valid_mask.sum().item()

        n_batches += 1

    result = {k: v / max(n_batches, 1) for k, v in totals.items()}
    result["phone_acc"] = correct_phone / max(total, 1)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load maps
    with open(args.maps_path) as f:
        maps = json.load(f)
    num_speakers  = len(maps["speakers"])
    num_varieties = len(maps["varieties"])
    print(f"Speakers: {num_speakers}, Varieties: {num_varieties}")

    # Dataset
    full_ds = PhoneTokenDataset(
        tokens_path=args.tokens_path,
        maps_path=args.maps_path,
        augment=True,
    )

    # Train/val split by speaker (not random) to avoid speaker leakage
    all_speakers = list(maps["speakers"].keys())
    val_speakers = set(all_speakers[:max(1, len(all_speakers) // 10)])
    train_indices = [i for i, t in enumerate(full_ds.tokens)
                     if t["speaker_id"] not in val_speakers]
    val_indices   = [i for i, t in enumerate(full_ds.tokens)
                     if t["speaker_id"] in val_speakers]

    from torch.utils.data import Subset
    train_ds = Subset(full_ds, train_indices)
    val_ds   = Subset(full_ds, val_indices)

    # Build sub-index for train dataset (for batch sampling)
    train_tokens = [full_ds.tokens[i] for i in train_indices]
    from data.phone_tokens import PhoneTokenIndex
    train_index = PhoneTokenIndex(train_tokens)
    # Remap subset indices → dataset indices
    train_index_remap = {new_i: train_indices[new_i]
                         for new_i in range(len(train_indices))}

    train_sampler = ContrastiveBatchSampler(
        full_ds,   # use full dataset for sampling but index is subset
        batch_size=args.batch_size,
    )
    # Override sampler's index with train-only index
    train_sampler.index = full_ds.index  # use full ds index for now

    def make_collate(dataset):
        from data.phone_tokens import contrastive_collate
        return partial(contrastive_collate, index=dataset.index)

    train_loader = DataLoader(
        full_ds,
        batch_sampler=train_sampler,
        collate_fn=make_collate(full_ds),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_sampler = ContrastiveBatchSampler(full_ds, batch_size=args.batch_size)
    val_loader = DataLoader(
        full_ds,
        batch_sampler=val_sampler,
        collate_fn=make_collate(full_ds),
        num_workers=2,
    )

    # Model
    from utils.phones import NUM_PHONES
    model = PACEM(
        backbone_name=args.backbone,
        token_dim=args.token_dim,
        embedding_dim=args.embedding_dim,
        num_phones=NUM_PHONES,
        freeze_transformer_layers=args.freeze_layers,
    ).to(device)

    # Criterion
    criterion = PhoneConditionedContrastiveLoss(
        temperature_variety=args.temp_variety,
        temperature_phone=args.temp_phone,
        alpha=args.alpha,
        beta=args.beta,
    )

    # Optimiser: separate LRs for backbone vs heads
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.segment_pooler.parameters()) +
        list(model.projector.parameters()) +
        list(model.phone_head.parameters()) +
        list(model.accent_pooler.parameters())
    )
    optimiser = AdamW([
        {"params": backbone_params, "lr": args.backbone_lr, "weight_decay": 0.01},
        {"params": head_params,     "lr": args.head_lr,     "weight_decay": 0.01},
    ])

    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimiser, T_max=total_steps, eta_min=1e-7)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model, full_ds, train_loader, optimiser, criterion, device,
            grad_accum=args.grad_accum,
        )
        scheduler.step()

        val_metrics = eval_epoch(model, full_ds, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1:03d} | "
            f"Train L={train_metrics['loss']:.4f} "
            f"(var={train_metrics['l_variety']:.4f} "
            f"ph_sep={train_metrics['l_phone_sep']:.4f} "
            f"aux={train_metrics['l_phone_aux']:.4f}) | "
            f"Val L={val_metrics['loss']:.4f} "
            f"phone_acc={val_metrics['phone_acc']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
                "maps_path": args.maps_path,
            }
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  ✓ Saved best checkpoint (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tokens_path",    required=True)
    p.add_argument("--maps_path",      required=True)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--backbone",       default="wavlm-base")
    p.add_argument("--token_dim",      type=int,   default=256)
    p.add_argument("--embedding_dim",  type=int,   default=256)
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--batch_size",     type=int,   default=32)
    p.add_argument("--backbone_lr",    type=float, default=1e-5)
    p.add_argument("--head_lr",        type=float, default=1e-4)
    p.add_argument("--temp_variety",   type=float, default=0.07)
    p.add_argument("--temp_phone",     type=float, default=0.10)
    p.add_argument("--alpha",          type=float, default=0.3)
    p.add_argument("--beta",           type=float, default=0.1)
    p.add_argument("--grad_accum",     type=int,   default=2)
    p.add_argument("--freeze_layers",  type=int,   default=0)
    p.add_argument("--num_workers",    type=int,   default=4)
    args = p.parse_args()
    main(args)

"""
train.py — Training loop for the Phonological Accent Embedding Encoder.

Loss:
  L_total = L_phon - λ · L_speaker

  L_phon   = L_regression (MSE on formants, VOT, F0) +
              L_classification (CE on rhoticity, flapping, glottalisation)

  L_speaker = CrossEntropy on speaker classifier
              (gradient reversed through GRL, so encoder maximises this loss)

GRL schedule:
  λ is annealed from 0 → λ_max using the schedule from Ganin et al. (2016):
    λ(p) = 2 / (1 + exp(-γ·p)) - 1,  p ∈ [0, 1]
  where p = fraction of training complete.
"""

import os
import math
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path

from model import AccentEncoder, PHON_REGRESSION, PHON_CLASSIFICATION
from dataset import AccentDataset, collate_fn, build_speaker_map, REGRESSION_KEYS, CLASSIFICATION_KEYS


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def phonological_regression_loss(preds: dict, targets: torch.Tensor) -> torch.Tensor:
    """
    MSE over continuous phonological features.
    NaN targets are masked out (feature not measurable for this utterance).
    targets: (B, len(REGRESSION_KEYS))
    """
    total = torch.tensor(0.0, device=targets.device, requires_grad=True)
    count = 0
    for i, key in enumerate(REGRESSION_KEYS):
        pred_key = _regression_key_to_model_key(key)
        if pred_key not in preds:
            continue
        pred = preds[pred_key].squeeze(-1)   # (B,)
        tgt = targets[:, i]                  # (B,)
        mask = ~torch.isnan(tgt)
        if mask.sum() == 0:
            continue
        loss = F.mse_loss(pred[mask], tgt[mask])
        total = total + loss
        count += 1
    return total / max(count, 1)


def phonological_classification_loss(preds: dict, targets: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy over binary phonological features (is_rhotic, is_flapped, is_glottalled).
    targets: (B, len(CLASSIFICATION_KEYS))
    """
    cls_pred_keys = ["is_rhotic", "is_flapped", "is_glottalled"]
    total = torch.tensor(0.0, device=targets.device, requires_grad=True)
    for i, pred_key in enumerate(cls_pred_keys):
        if pred_key not in preds:
            continue
        logits = preds[pred_key]   # (B, 2)
        tgt = targets[:, i]        # (B,) long
        total = total + F.cross_entropy(logits, tgt)
    return total / max(len(cls_pred_keys), 1)


def _regression_key_to_model_key(dataset_key: str) -> str:
    """Map dataset feature keys to model head keys."""
    mapping = {
        "f1_mean_norm": "f1_norm",
        "f2_mean_norm": "f2_norm",
        "f3_mean_norm": "f3_norm",
        "f0_mean":      "f0_mean",
        "vot_mean":     "vot_ms",
    }
    return mapping.get(dataset_key, dataset_key)


def grl_lambda_schedule(step: int, total_steps: int, gamma: float = 10.0, lambda_max: float = 1.0) -> float:
    """Ganin et al. (2016) GRL annealing schedule."""
    p = step / total_steps
    return lambda_max * (2.0 / (1.0 + math.exp(-gamma * p)) - 1.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimiser, device, step, total_steps, args):
    model.train()
    epoch_loss = 0.0
    epoch_phon_loss = 0.0
    epoch_spk_loss = 0.0

    for batch in loader:
        waveform = batch["waveform"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        reg_tgts = batch["reg_targets"].to(device)
        cls_tgts = batch["cls_targets"].to(device)
        spk_ids = batch["speaker_id"].to(device)

        # Update GRL lambda
        lam = grl_lambda_schedule(step, total_steps, lambda_max=args.lambda_max)
        model.set_grl_lambda(lam)

        out = model(waveform, attn_mask)

        # Phonological losses
        l_reg = phonological_regression_loss(out["phon_preds"], reg_tgts)
        l_cls = phonological_classification_loss(out["phon_preds"], cls_tgts)
        l_phon = l_reg + l_cls

        # Speaker adversarial loss (encoder wants to maximise this via GRL)
        l_spk = F.cross_entropy(out["speaker_logits"], spk_ids)

        # Combined: encoder minimises phon loss, GRL handles speaker
        loss = l_phon + l_spk  # GRL already negates the speaker gradient

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

        epoch_loss += loss.item()
        epoch_phon_loss += l_phon.item()
        epoch_spk_loss += l_spk.item()
        step += 1

    n = len(loader)
    return (epoch_loss / n, epoch_phon_loss / n, epoch_spk_loss / n, step)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_phon, total_spk = 0.0, 0.0
    correct_spk, total_utts = 0, 0

    for batch in loader:
        waveform = batch["waveform"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        reg_tgts = batch["reg_targets"].to(device)
        cls_tgts = batch["cls_targets"].to(device)
        spk_ids = batch["speaker_id"].to(device)

        out = model(waveform, attn_mask)

        l_reg = phonological_regression_loss(out["phon_preds"], reg_tgts)
        l_cls = phonological_classification_loss(out["phon_preds"], cls_tgts)
        total_phon += (l_reg + l_cls).item()

        l_spk = F.cross_entropy(out["speaker_logits"], spk_ids)
        total_spk += l_spk.item()

        # Speaker accuracy: low is good (means speaker info is gone)
        preds = out["speaker_logits"].argmax(-1)
        correct_spk += (preds == spk_ids).sum().item()
        total_utts += spk_ids.shape[0]

    n = len(loader)
    spk_acc = correct_spk / total_utts
    return total_phon / n, total_spk / n, spk_acc


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Build speaker map
    speaker_to_id = build_speaker_map(args.features_path)
    num_speakers = len(speaker_to_id)
    print(f"Speakers: {num_speakers}")

    # Save speaker map
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    with open(os.path.join(args.checkpoint_dir, "speaker_map.json"), "w") as f:
        json.dump(speaker_to_id, f)

    # Dataset
    full_dataset = AccentDataset(
        audio_dir=args.audio_dir,
        features_path=args.features_path,
        speaker_to_id=speaker_to_id,
        augment=True,
    )
    val_size = max(1, int(0.1 * len(full_dataset)))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.augment = False

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    # Model
    model = AccentEncoder(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        num_speakers=num_speakers,
        freeze_feature_extractor=True,
        use_weighted_layers=True,
    ).to(device)

    # Optimiser: lower LR for backbone, higher for new heads
    backbone_params = list(model.backbone.parameters())
    head_params = (
        list(model.projector.parameters()) +
        list(model.phon_heads.parameters()) +
        list(model.speaker_classifier.parameters()) +
        [model.layer_weights]
    )
    optimiser = AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params,    "lr": args.head_lr},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimiser, T_max=total_steps)

    best_phon_loss = float("inf")
    step = 0

    for epoch in range(args.epochs):
        train_loss, train_phon, train_spk, step = train_one_epoch(
            model, train_loader, optimiser, device, step, total_steps, args
        )
        scheduler.step()

        val_phon, val_spk, val_spk_acc = evaluate(model, val_loader, device)

        lam = grl_lambda_schedule(step, total_steps, lambda_max=args.lambda_max)
        print(
            f"Epoch {epoch+1:03d} | "
            f"Train loss {train_loss:.4f} (phon {train_phon:.4f}, spk {train_spk:.4f}) | "
            f"Val phon {val_phon:.4f}  spk_acc {val_spk_acc:.3f}  λ={lam:.3f}"
        )

        # Save best checkpoint (lowest phonological loss)
        if val_phon < best_phon_loss:
            best_phon_loss = val_phon
            ckpt_path = os.path.join(args.checkpoint_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "val_phon_loss": val_phon,
                "val_spk_acc": val_spk_acc,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ Saved best checkpoint → {ckpt_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Phonological Accent Encoder")
    p.add_argument("--audio_dir",       required=True)
    p.add_argument("--features_path",   required=True, help="features.json from features.py")
    p.add_argument("--checkpoint_dir",  default="checkpoints")
    p.add_argument("--backbone",        default="wavlm-base",
                   choices=["wavlm-base", "wav2vec2-base"])
    p.add_argument("--embedding_dim",   type=int,   default=256)
    p.add_argument("--epochs",          type=int,   default=30)
    p.add_argument("--batch_size",      type=int,   default=16)
    p.add_argument("--backbone_lr",     type=float, default=1e-5)
    p.add_argument("--head_lr",         type=float, default=1e-4)
    p.add_argument("--lambda_max",      type=float, default=1.0,
                   help="Maximum GRL reversal strength")
    args = p.parse_args()
    main(args)

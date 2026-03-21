"""
WoundScope v2 – Improved training script for L4 GPU.

Improvements over v1:
  - ConvNeXt-Small backbone (stronger than EfficientNet-B0)
  - MixUp + CutMix augmentation (fixes overfitting)
  - Weighted random sampler (fixes class imbalance)
  - Label smoothing
  - AdamW + cosine warmup scheduler

Usage (on L4):
    python src/train_v2.py --data_csv dataset/labels.csv \
                            --img_root dataset/wound_images \
                            --epochs 30 --batch_size 64
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm

from data_loader import (
    build_dataloaders, WoundDataset, WOUND_CLASSES, NUM_LOCATIONS,
    TRAIN_TRANSFORM, VAL_TRANSFORM
)
from utils import evaluate, print_report, save_checkpoint, get_device, plot_training_curves

import pandas as pd
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class WoundScopeV2(nn.Module):
    """ConvNeXt-Small backbone + location embedding → 4-class classifier."""

    def __init__(self, arch="convnext_small", num_classes=4,
                 num_locations=NUM_LOCATIONS, loc_emb_dim=16,
                 hidden_dim=256, dropout=0.4):
        super().__init__()

        self.image_branch = timm.create_model(arch, pretrained=True, num_classes=0)
        self.img_feat_dim = self.image_branch.num_features

        self.loc_embedding = nn.Embedding(num_locations, loc_emb_dim)

        fused_dim = self.img_feat_dim + loc_emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, img, loc_idx):
        img_feat = self.image_branch(img)
        loc_feat = self.loc_embedding(loc_idx)
        return self.classifier(torch.cat([img_feat, loc_feat], dim=1))

    @property
    def conv_head(self):
        # For Grad-CAM compatibility
        return self.image_branch.head.norm


# ──────────────────────────────────────────────────────────────────────────────
# Weighted sampler (fix class imbalance)
# ──────────────────────────────────────────────────────────────────────────────

def make_weighted_sampler(dataset):
    labels = [WOUND_CLASSES.index(dataset.df.iloc[i]["wound_type"]) for i in range(len(dataset))]
    class_counts = np.bincount(labels, minlength=len(WOUND_CLASSES))
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ──────────────────────────────────────────────────────────────────────────────
# MixUp / CutMix
# ──────────────────────────────────────────────────────────────────────────────

def mixup(imgs, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    mixed = lam * imgs + (1 - lam) * imgs[idx]
    return mixed, labels, labels[idx], lam


def cutmix(imgs, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    B, C, H, W = imgs.shape

    cut_rat = np.sqrt(1 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)

    mixed = imgs.clone()
    mixed[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, labels, labels[idx], lam


def mixup_cutmix_loss(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, use_mix=True):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, locs, labels in loader:
        imgs, locs, labels = imgs.to(device), locs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_mix and np.random.rand() > 0.5:
            if np.random.rand() > 0.5:
                imgs, labels_a, labels_b, lam = mixup(imgs, labels)
            else:
                imgs, labels_a, labels_b, lam = cutmix(imgs, labels)
            logits = model(imgs, locs)
            loss = mixup_cutmix_loss(criterion, logits, labels_a, labels_b, lam)
            correct += (lam * (logits.argmax(1) == labels_a).float() +
                        (1 - lam) * (logits.argmax(1) == labels_b).float()).sum().item()
        else:
            logits = model(imgs, locs)
            loss = criterion(logits, labels)
            correct += (logits.argmax(1) == labels).sum().item()

        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total += imgs.size(0)

    return total_loss / total, correct / total


def val_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, locs, labels in loader:
            imgs, locs, labels = imgs.to(device), locs.to(device), labels.to(device)
            logits = model(imgs, locs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total


# ──────────────────────────────────────────────────────────────────────────────
# Dataloaders with weighted sampler
# ──────────────────────────────────────────────────────────────────────────────

def build_loaders_v2(csv_path, img_root, batch_size, seed=42):
    from data_loader import BODY_LOCATIONS
    df = pd.read_csv(csv_path)
    if "location_idx" not in df.columns:
        df["location_idx"] = df["location"].apply(
            lambda x: BODY_LOCATIONS.index(x) if x in BODY_LOCATIONS else 0
        )

    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["wound_type"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["wound_type"], random_state=seed
    )

    train_ds = WoundDataset(train_df, img_root, TRAIN_TRANSFORM)
    val_ds   = WoundDataset(val_df,   img_root, VAL_TRANSFORM)
    test_ds  = WoundDataset(test_df,  img_root, VAL_TRANSFORM)

    sampler = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = get_device()
    train_loader, val_loader, test_loader = build_loaders_v2(
        args.data_csv, args.img_root, batch_size=args.batch_size
    )

    model = WoundScopeV2(arch=args.arch, num_classes=len(WOUND_CLASSES)).to(device)
    print(f"Backbone: {args.arch}  |  Feature dim: {model.img_feat_dim}")

    # Label smoothing helps with overconfident predictions
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1: head only
    print(f"\n--- Phase 1: Head-only (3 epochs) ---")
    for p in model.image_branch.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4
    )
    for ep in range(3):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device, use_mix=False)
        vl, va = val_one_epoch(model, val_loader, criterion, device)
        print(f"  Epoch {ep+1}/3  train_acc={ta:.3f}  val_acc={va:.3f}")

    # Phase 2: full fine-tune with mix augmentation
    print(f"\n--- Phase 2: Full fine-tune ({args.epochs} epochs) ---")
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # Cosine schedule with linear warmup
    def lr_lambda(ep):
        warmup = 3
        if ep < warmup:
            return ep / warmup
        progress = (ep - warmup) / (args.epochs - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    for ep in range(args.epochs):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device, use_mix=True)
        vl, va = val_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tl)
        val_losses.append(vl)
        val_accs.append(va)

        print(f"Epoch {ep+1:3d}/{args.epochs}  train_loss={tl:.4f} train_acc={ta:.3f}  val_loss={vl:.4f} val_acc={va:.3f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if va > best_val_acc:
            best_val_acc = va
            save_checkpoint(
                {"model_state": model.state_dict(), "arch": args.arch, "epoch": ep + 1},
                "models/multimodal_model.pth"
            )
            print(f"  ✓ Saved (val_acc={va:.3f})")

    plot_training_curves(train_losses, val_losses, val_accs, "outputs/v2_curves.png")

    # Test
    print("\n--- Test Set Evaluation ---")
    ckpt = torch.load("models/multimodal_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    def model_fn(imgs, locs):
        return model(imgs, locs)

    acc, f1, preds, labels = evaluate(model_fn, test_loader, device)
    print(f"Test Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
    print_report(preds, labels, out_path="outputs/eval_multimodal.txt")
    print_report(preds, labels, out_path="outputs/metrics_report.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv",   default="dataset/labels.csv")
    parser.add_argument("--img_root",   default="dataset/wound_images")
    parser.add_argument("--arch",       default="convnext_small")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=5e-5)
    args = parser.parse_args()
    main(args)

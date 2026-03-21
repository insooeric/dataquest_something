"""
WoundScope v3 – ViT-Small + location embedding, multi-task learning.

Wound-type classification (7 classes) + severity grading (4-level ordinal)
via a shared backbone and two separate output heads.

Usage:
    python3 src/train_v3.py --data_csv dataset/labels.csv \\
                             --img_root dataset/wound_images \\
                             --epochs 100 --batch_size 32
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
import torchvision.transforms as T
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_loader import (
    WoundDataset, WOUND_CLASSES, BODY_LOCATIONS, NUM_LOCATIONS, NUM_SEVERITY,
    SEVERITY_UNKNOWN, VAL_TRANSFORM,
)
from utils import (
    evaluate, print_report, plot_confusion_matrix,
    save_checkpoint, get_device, plot_training_curves,
)


# ── Augmentation ────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM_V3 = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])


# ── Model ───────────────────────────────────────────────────────────────────────

class WoundScope(nn.Module):
    """
    ViT-Small backbone + location embedding.
    Two output heads:
      - wound_head   → (B, num_classes)   wound-type logits
      - severity_head → (B, num_severity)  severity logits
    """

    def __init__(self, arch="vit_small_patch16_224",
                 num_classes=len(WOUND_CLASSES),
                 num_locations=NUM_LOCATIONS,
                 loc_emb_dim=32,
                 hidden_dim=256,
                 dropout=0.4,
                 num_severity=NUM_SEVERITY):
        super().__init__()

        self.backbone    = timm.create_model(arch, pretrained=True, num_classes=0)
        self.feat_dim    = self.backbone.num_features   # 384 for vit_small

        self.loc_embedding = nn.Embedding(num_locations, loc_emb_dim)

        fused_dim = self.feat_dim + loc_emb_dim
        self.shared = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.wound_head    = nn.Linear(hidden_dim, num_classes)
        self.severity_head = nn.Linear(hidden_dim, num_severity)

    def forward(self, img, loc_idx):
        img_feat = self.backbone(img)                           # (B, feat_dim)
        loc_feat = self.loc_embedding(loc_idx)                  # (B, loc_emb_dim)
        shared   = self.shared(torch.cat([img_feat, loc_feat], dim=1))
        return self.wound_head(shared), self.severity_head(shared)


# ── Weighted sampler ────────────────────────────────────────────────────────────

def make_weighted_sampler(dataset):
    labels      = [WOUND_CLASSES.index(dataset.df.iloc[i]["wound_type"]) for i in range(len(dataset))]
    class_counts = np.bincount(labels, minlength=len(WOUND_CLASSES))
    weights      = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── MixUp / CutMix ─────────────────────────────────────────────────────────────

def mixup(imgs, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1 - lam) * imgs[idx], labels, labels[idx], lam


def cutmix(imgs, labels, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    B, C, H, W = imgs.shape
    cut_rat = np.sqrt(1 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0); x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0); y2 = min(cy + cut_h // 2, H)
    mixed = imgs.clone()
    mixed[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
    lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
    return mixed, labels, labels[idx], lam


def mix_loss(criterion, logits, a, b, lam):
    return lam * criterion(logits, a) + (1 - lam) * criterion(logits, b)


# ── Loss ────────────────────────────────────────────────────────────────────────

def multitask_loss(wound_logits, sev_logits, wound_labels, sev_labels,
                   wound_criterion, sev_criterion, sev_weight=0.3):
    wound_loss = wound_criterion(wound_logits, wound_labels)

    sev_mask = sev_labels >= 0
    if sev_mask.any():
        sev_loss = sev_criterion(sev_logits[sev_mask], sev_labels[sev_mask])
        return wound_loss + sev_weight * sev_loss, wound_loss
    return wound_loss, wound_loss


# ── Dataloaders ─────────────────────────────────────────────────────────────────

def build_loaders(csv_path, img_root, batch_size, seed=42):
    df = pd.read_csv(csv_path)
    if "location_idx" not in df.columns:
        df["location_idx"] = df["location"].apply(
            lambda x: BODY_LOCATIONS.index(x) if x in BODY_LOCATIONS else 0
        )
    if "severity" not in df.columns:
        df["severity"] = SEVERITY_UNKNOWN

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["wound_type"], random_state=seed)
    val_df,   test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["wound_type"], random_state=seed)

    train_ds = WoundDataset(train_df, img_root, TRAIN_TRANSFORM_V3)
    val_ds   = WoundDataset(val_df,   img_root, VAL_TRANSFORM)
    test_ds  = WoundDataset(test_df,  img_root, VAL_TRANSFORM)

    sampler      = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,   num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,   num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


# ── Training loops ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, wound_criterion, sev_criterion, device, sev_weight=0.3):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    bar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for imgs, locs, labels, severity in bar:
        imgs, locs, labels, severity = (
            imgs.to(device), locs.to(device), labels.to(device), severity.to(device)
        )
        optimizer.zero_grad()

        if np.random.rand() > 0.5:
            if np.random.rand() > 0.5:
                imgs, la, lb, lam = mixup(imgs, labels)
            else:
                imgs, la, lb, lam = cutmix(imgs, labels)
            wound_logits, _ = model(imgs, locs)
            loss = mix_loss(wound_criterion, wound_logits, la, lb, lam)
            correct += (
                lam * (wound_logits.argmax(1) == la).float() +
                (1 - lam) * (wound_logits.argmax(1) == lb).float()
            ).sum().item()
        else:
            wound_logits, sev_logits = model(imgs, locs)
            loss, _ = multitask_loss(
                wound_logits, sev_logits, labels, severity,
                wound_criterion, sev_criterion, sev_weight,
            )
            correct += (wound_logits.argmax(1) == labels).sum().item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
        bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")

    return total_loss / total, correct / total


def val_one_epoch(model, loader, wound_criterion, sev_criterion, device, sev_weight=0.3):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        bar = tqdm(loader, desc="    val", leave=False, unit="batch")
        for imgs, locs, labels, severity in bar:
            imgs, locs, labels, severity = (
                imgs.to(device), locs.to(device), labels.to(device), severity.to(device)
            )
            wound_logits, sev_logits = model(imgs, locs)
            loss, _ = multitask_loss(
                wound_logits, sev_logits, labels, severity,
                wound_criterion, sev_criterion, sev_weight,
            )
            total_loss += loss.item() * imgs.size(0)
            correct    += (wound_logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
            bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")
    return total_loss / total, correct / total


# ── Main ────────────────────────────────────────────────────────────────────────

def main(args):
    device = get_device()
    train_loader, val_loader, test_loader = build_loaders(
        args.data_csv, args.img_root, args.batch_size
    )

    model = WoundScope(arch=args.arch, num_classes=len(WOUND_CLASSES)).to(device)
    print(f"Backbone: {args.arch}  |  Feature dim: {model.feat_dim}  |  Classes: {len(WOUND_CLASSES)}")

    wound_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    sev_criterion   = nn.CrossEntropyLoss()

    # Phase 1: warm up classifier heads only (3 epochs)
    print("\n--- Phase 1: Head warm-up (3 epochs) ---")
    for p in model.backbone.parameters():
        p.requires_grad = False
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4
    )
    for ep in range(3):
        tl, ta = train_one_epoch(model, train_loader, optimizer, wound_criterion, sev_criterion, device)
        vl, va = val_one_epoch(model, val_loader, wound_criterion, sev_criterion, device)
        print(f"  Epoch {ep+1}/3  train_acc={ta:.3f}  val_acc={va:.3f}")

    # Phase 2: fine-tune all layers
    print(f"\n--- Phase 2: Full fine-tune ({args.epochs} epochs, patience={args.patience}) ---")
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(ep):
        warmup = 5
        if ep < warmup:
            return ep / warmup
        progress = (ep - warmup) / max(args.epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_acc = 0.0
    no_improve   = 0
    train_losses, val_losses, val_accs = [], [], []
    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    for ep in range(args.epochs):
        t0 = time.time()
        tl, ta = train_one_epoch(model, train_loader, optimizer, wound_criterion, sev_criterion, device)
        vl, va = val_one_epoch(model, val_loader, wound_criterion, sev_criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        train_losses.append(tl)
        val_losses.append(vl)
        val_accs.append(va)

        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {ep+1:3d}/{args.epochs}  "
              f"train_loss={tl:.4f} train_acc={ta:.3f}  "
              f"val_loss={vl:.4f} val_acc={va:.3f}  "
              f"lr={lr_now:.2e}  {elapsed:.0f}s")

        if va > best_val_acc:
            best_val_acc = va
            no_improve   = 0
            save_checkpoint(
                {"model_state": model.state_dict(), "arch": args.arch,
                 "epoch": ep + 1, "version": "v3", "num_classes": len(WOUND_CLASSES)},
                "models/woundscope_v3.pth",
            )
            print(f"  ✓ New best (val_acc={va:.3f})")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {ep+1} (no improvement for {args.patience} epochs)")
                break

    plot_training_curves(train_losses, val_losses, val_accs, "outputs/v3_curves.png")

    # ── Test set evaluation ─────────────────────────────────────────────────────
    print("\n--- Test Set Evaluation ---")
    ckpt = torch.load("models/woundscope_v3.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    def model_fn(imgs, locs):
        return model(imgs, locs)

    acc, f1, preds, labels, sev_acc = evaluate(model_fn, test_loader, device)
    print(f"Test Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}", end="")
    if sev_acc is not None:
        print(f"  Severity Acc: {sev_acc:.4f}", end="")
    print()

    # Collect probs + severity preds for full report
    all_probs, all_sev_preds, all_sev_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, locs, lbl, sev in test_loader:
            imgs, locs = imgs.to(device), locs.to(device)
            w_logits, s_logits = model(imgs, locs)
            all_probs.append(F.softmax(w_logits, dim=1).cpu().numpy())
            s_preds = s_logits.argmax(dim=1).cpu().numpy()
            s_true  = sev.numpy()
            mask    = s_true >= 0
            if mask.any():
                all_sev_preds.extend(s_preds[mask].tolist())
                all_sev_labels.extend(s_true[mask].tolist())

    probs_arr = np.vstack(all_probs) if all_probs else None
    print_report(
        preds, labels,
        probs       = probs_arr,
        sev_preds   = all_sev_preds   if all_sev_labels else None,
        sev_labels  = all_sev_labels  if all_sev_labels else None,
        out_path    = "outputs/eval_v3.txt",
    )
    plot_confusion_matrix(preds, labels, out_path="outputs/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv",   default="dataset/labels.csv")
    parser.add_argument("--img_root",   default="dataset/wound_images")
    parser.add_argument("--arch",       default="vit_small_patch16_224")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--patience",   type=int,   default=20)
    args = parser.parse_args()
    main(args)

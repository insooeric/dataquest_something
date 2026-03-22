"""
WoundScope – Phase 1: Pretrain backbone from scratch.

Trains WoundCNN on wound images with a simple classification head.
No pretrained weights. No timm. No external model zoo.

Saves: models/backbone_pretrained.pth  (backbone weights only)

Usage:
    python src_pretrain/pretrain.py
    python src_pretrain/pretrain.py --epochs 120 --batch_size 64 --lr 3e-4
"""

import argparse
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))           # src_pretrain/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src_finetuning"))  # data_loader

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from architecture import WoundCNN
from data_loader import (
    WoundDataset, WOUND_CLASSES, BODY_LOCATIONS, SEVERITY_UNKNOWN,
)


# ── Augmentation ────────────────────────────────────────────────────────────────

TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.RandomResizedCrop(224, scale=(0.65, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.35, hue=0.06),
    T.RandomGrayscale(p=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    T.RandomErasing(p=0.2, scale=(0.02, 0.15)),
])

VAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Pretrain model (backbone + simple linear head) ───────────────────────────────

class PretrainModel(nn.Module):
    """WoundCNN backbone + single linear classification head for pretraining."""

    def __init__(self, num_classes=len(WOUND_CLASSES)):
        super().__init__()
        self.backbone = WoundCNN()
        self.head     = nn.Linear(WoundCNN.FEAT_DIM, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


# ── Data ─────────────────────────────────────────────────────────────────────────

import re

def _stratify(frame):
    counts = frame["wound_type"].value_counts()
    return frame["wound_type"] if (counts >= 2).all() else None


def build_loaders(csv_path, img_root, batch_size, seed=42):
    df = pd.read_csv(csv_path)
    if "severity" not in df.columns:
        df["severity"] = SEVERITY_UNKNOWN
    if "location_idx" not in df.columns:
        df["location_idx"] = 0

    df["_base"] = df["image_path"].apply(
        lambda p: re.sub(r"_\d+\.(jpg|jpeg|png)$", "", p.replace("\\", "/"), flags=re.IGNORECASE)
    )
    bases = df[["_base", "wound_type"]].drop_duplicates("_base")

    train_bases, val_bases = train_test_split(
        bases, test_size=0.15, stratify=_stratify(bases), random_state=seed
    )
    train_df = df[df["_base"].isin(train_bases["_base"])].drop(columns="_base")
    val_df   = df[df["_base"].isin(val_bases["_base"])].drop(columns="_base")

    train_ds = WoundDataset(train_df, img_root, TRAIN_TRANSFORM)
    val_ds   = WoundDataset(val_df,   img_root, VAL_TRANSFORM)

    labels       = train_ds.df["wound_type"].map(WOUND_CLASSES.index).to_numpy()
    class_counts = np.bincount(labels, minlength=len(WOUND_CLASSES))
    weights      = 1.0 / class_counts[labels]
    sampler      = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,    num_workers=4, pin_memory=True)
    return train_loader, val_loader


# ── MixUp ────────────────────────────────────────────────────────────────────────

def mixup(imgs, labels, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1 - lam) * imgs[idx], labels, labels[idx], lam


# ── Train / val loops ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    bar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for imgs, locs, labels, _ in bar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        if np.random.rand() > 0.5:
            imgs, la, lb, lam = mixup(imgs, labels)
            logits = model(imgs)
            loss   = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
            correct += (lam * (logits.argmax(1) == la).float()
                        + (1 - lam) * (logits.argmax(1) == lb).float()).sum().item()
        else:
            logits = model(imgs)
            loss   = criterion(logits, labels)
            correct += (logits.argmax(1) == labels).sum().item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        total      += imgs.size(0)
        bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")
    return total_loss / total, correct / total


def val_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        bar = tqdm(loader, desc="    val", leave=False, unit="batch")
        for imgs, locs, labels, _ in bar:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += imgs.size(0)
            bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{correct/total:.3f}")
    return total_loss / total, correct / total


# ── Plotting ─────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _plot_curves(train_losses, val_losses, val_accs, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses,   label="Val Loss")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Pretrain Loss Curves"); ax1.legend()
    ax2.plot(val_accs, color="green", label="Val Accuracy")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Pretrain Val Accuracy"); ax2.legend()
    plt.tight_layout()
    plt.savefig(out_path); plt.close()


# ── Main ─────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        d = torch.device("cpu")
        print("No GPU found, using CPU.")
    return d


def main(args):
    device = get_device()
    train_loader, val_loader = build_loaders(args.data_csv, args.img_root, args.batch_size)

    model     = PretrainModel(num_classes=len(WOUND_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(ep):
        warmup = 8
        if ep < warmup:
            return (ep + 1) / warmup
        progress = (ep - warmup) / max(args.epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler  = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_acc   = 0.0
    no_improve = 0
    train_losses, val_losses, val_accs = [], [], []

    os.makedirs("models",  exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    print(f"\n--- Pretraining WoundCNN from scratch ({args.epochs} epochs) ---")

    for ep in range(args.epochs):
        t0     = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = val_one_epoch(model, val_loader, criterion, device)
        scheduler.step()
        train_losses.append(tl); val_losses.append(vl); val_accs.append(va)
        print(f"Epoch {ep+1:3d}/{args.epochs}  "
              f"train_loss={tl:.4f} train_acc={ta:.3f}  "
              f"val_loss={vl:.4f} val_acc={va:.3f}  "
              f"lr={lr_now:.2e}  {time.time()-t0:.0f}s")

        if va > best_acc:
            best_acc   = va
            no_improve = 0
            torch.save({"backbone_state": model.backbone.state_dict(),
                        "epoch": ep + 1, "val_acc": va},
                       "models/backbone_pretrained.pth")
            print(f"  ✓ New best (val_acc={va:.3f}) — backbone saved")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {ep+1}")
                break

    _plot_curves(train_losses, val_losses, val_accs, "outputs/pretrain_curves.png")
    print(f"\nPretraining complete. Best val_acc: {best_acc:.3f}")
    print("Backbone saved to:      models/backbone_pretrained.pth")
    print("Training curves saved:  outputs/pretrain_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv",   default="dataset/labels.csv")
    parser.add_argument("--img_root",   default="dataset/wound_images")
    parser.add_argument("--epochs",     type=int,   default=120)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--patience",   type=int,   default=20)
    args = parser.parse_args()
    main(args)

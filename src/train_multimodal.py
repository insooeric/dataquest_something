"""
WoundScope – Multimodal model: CNN image features + learnable location embedding.

Usage:
    python src/train_multimodal.py --data_csv dataset/labels.csv \
                                    --img_root dataset/wound_images \
                                    --baseline_ckpt models/baseline_model.pth \
                                    --epochs 15 --lr 5e-5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torchvision.models as models
import timm

from data_loader import build_dataloaders, WOUND_CLASSES, NUM_LOCATIONS
from utils import evaluate, print_report, save_checkpoint, get_device, plot_training_curves


# ──────────────────────────────────────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────────────────────────────────────

class WoundScopeMultimodal(nn.Module):
    """
    Image branch (ResNet50 backbone, no final FC) + location embedding.
    Concatenated features → MLP → 4-class output.
    """

    def __init__(self, arch="resnet50", num_classes=4, num_locations=NUM_LOCATIONS,
                 loc_emb_dim=16, hidden_dim=256, dropout=0.3):
        super().__init__()

        # Image branch
        if arch == "resnet50":
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.img_feat_dim = backbone.fc.in_features  # 2048
            backbone.fc = nn.Identity()
            self.image_branch = backbone
        elif arch == "efficientnet_b0":
            backbone = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
            self.img_feat_dim = backbone.num_features
            self.image_branch = backbone
        else:
            raise ValueError(f"Unknown arch: {arch}")

        # Location branch
        self.loc_embedding = nn.Embedding(num_locations, loc_emb_dim)

        # Fusion classifier
        fused_dim = self.img_feat_dim + loc_emb_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, img, loc_idx):
        img_feat = self.image_branch(img)           # (B, img_feat_dim)
        loc_feat = self.loc_embedding(loc_idx)      # (B, loc_emb_dim)
        x = torch.cat([img_feat, loc_feat], dim=1)  # (B, fused_dim)
        return self.classifier(x)

    def get_image_backbone(self):
        return self.image_branch


# ──────────────────────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, locs, labels in loader:
        imgs, locs, labels = imgs.to(device), locs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs, locs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
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


def main(args):
    device = get_device()
    train_loader, val_loader, test_loader = build_dataloaders(
        args.data_csv, args.img_root, batch_size=args.batch_size
    )

    model = WoundScopeMultimodal(arch=args.arch, num_classes=len(WOUND_CLASSES)).to(device)

    # Warm-start image backbone from baseline checkpoint
    if args.baseline_ckpt and os.path.exists(args.baseline_ckpt):
        print(f"Loading baseline weights from {args.baseline_ckpt}")
        ckpt = torch.load(args.baseline_ckpt, map_location=device)
        state = ckpt["model_state"]
        # Only load matching keys (skip final FC which changed)
        backbone_state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        missing, unexpected = model.image_branch.load_state_dict(backbone_state, strict=False)
        print(f"  Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    for ep in range(args.epochs):
        tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl, va = val_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(tl)
        val_losses.append(vl)
        val_accs.append(va)

        print(f"Epoch {ep+1:3d}/{args.epochs}  train_loss={tl:.4f} train_acc={ta:.3f}  val_loss={vl:.4f} val_acc={va:.3f}")

        if va > best_val_acc:
            best_val_acc = va
            save_checkpoint({
                "model_state": model.state_dict(),
                "arch": args.arch,
                "epoch": ep + 1,
            }, "models/multimodal_model.pth")

    plot_training_curves(train_losses, val_losses, val_accs, "outputs/multimodal_curves.png")

    # Test evaluation
    print("\n--- Test Set Evaluation ---")
    best_ckpt = torch.load("models/multimodal_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    def model_fn(imgs, locs):
        return model(imgs, locs)

    acc, f1, preds, labels = evaluate(model_fn, test_loader, device)
    print(f"Test Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
    print_report(preds, labels, out_path="outputs/eval_multimodal.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default="dataset/labels.csv")
    parser.add_argument("--img_root", default="dataset/wound_images")
    parser.add_argument("--arch", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--baseline_ckpt", default="models/baseline_model.pth")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    args = parser.parse_args()
    main(args)

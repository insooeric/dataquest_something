"""
WoundScope – Baseline image-only CNN (ResNet50 / EfficientNet-B0).

Usage:
    python src/train_baseline.py --data_csv dataset/labels.csv \
                                  --img_root dataset/wound_images \
                                  --epochs 20 --batch_size 32 --lr 1e-4
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torchvision.models as models
import timm

from data_loader import build_dataloaders, WOUND_CLASSES
from utils import evaluate, print_report, save_checkpoint, get_device, plot_training_curves


def build_model(arch="resnet50", num_classes=4):
    if arch == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch == "efficientnet_b0":
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return model


def train_one_epoch(model, loader, optimizer, criterion, device, freeze_backbone=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, locs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
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
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
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

    model = build_model(args.arch, num_classes=len(WOUND_CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()

    # Phase 1: freeze backbone, train only head
    if args.freeze_epochs > 0:
        print(f"\n--- Phase 1: Frozen backbone for {args.freeze_epochs} epochs ---")
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name and "head" not in name:
                param.requires_grad = False
        head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(head_params, lr=args.lr * 10)
        for ep in range(args.freeze_epochs):
            tl, ta = train_one_epoch(model, train_loader, optimizer, criterion, device)
            vl, va = val_one_epoch(model, val_loader, criterion, device)
            print(f"  Epoch {ep+1}/{args.freeze_epochs}  train_loss={tl:.4f} train_acc={ta:.3f}  val_loss={vl:.4f} val_acc={va:.3f}")

        # Unfreeze all
        for p in model.parameters():
            p.requires_grad = True

    # Phase 2: fine-tune all layers
    print(f"\n--- Phase 2: Fine-tuning all layers for {args.epochs} epochs ---")
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
            save_checkpoint({"model_state": model.state_dict(), "arch": args.arch, "epoch": ep+1},
                            "models/baseline_model.pth")

    plot_training_curves(train_losses, val_losses, val_accs, "outputs/baseline_curves.png")

    # Test evaluation
    print("\n--- Test Set Evaluation ---")
    best_ckpt = torch.load("models/baseline_model.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state"])

    def model_fn(imgs, locs=None):
        return model(imgs)

    acc, f1, preds, labels = evaluate(model_fn, test_loader, device)
    print(f"Test Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
    print_report(preds, labels, out_path="outputs/eval_baseline.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", default="dataset/labels.csv")
    parser.add_argument("--img_root", default="dataset/wound_images")
    parser.add_argument("--arch", default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--freeze_epochs", type=int, default=2,
                        help="Epochs to train only head before unfreezing (0 to skip)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    main(args)

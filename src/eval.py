"""
WoundScope – Standalone evaluation script.

Loads a trained checkpoint, runs test set evaluation, and saves:
  outputs/eval_v3.txt
  outputs/confusion_matrix.png
  outputs/v3_curves.png  (if training history is available in the checkpoint)

Usage:
    python src/eval.py
    python src/eval.py --ckpt models/woundscope_v3.pth --batch_size 64
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch

from data_loader import WOUND_CLASSES
from train import WoundScope, build_loaders
from utils import (
    evaluate, print_report, plot_confusion_matrix,
    get_device, plot_training_curves,
)


def main(args):
    device = get_device()

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    arch         = ckpt.get("arch", "vit_small_patch16_224")
    num_classes  = ckpt.get("num_classes", len(WOUND_CLASSES))

    model = WoundScope(arch=arch, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}  (epoch {ckpt.get('epoch', '?')}, arch={arch})")

    _, _, test_loader = build_loaders(args.data_csv, args.img_root, args.batch_size)

    def model_fn(imgs, locs):
        return model(imgs, locs)

    os.makedirs("outputs", exist_ok=True)

    print("\n--- Test Set Evaluation ---")
    acc, f1, preds, labels, sev_acc, probs_arr, all_sev_preds, all_sev_labels = evaluate(
        model_fn, test_loader, device
    )
    print(f"Test Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}", end="")
    if sev_acc is not None:
        print(f"  Severity Acc: {sev_acc:.4f}", end="")
    print()

    print_report(
        preds, labels,
        probs      = probs_arr,
        sev_preds  = all_sev_preds  if all_sev_labels else None,
        sev_labels = all_sev_labels if all_sev_labels else None,
        out_path   = "outputs/eval_v3.txt",
    )
    plot_confusion_matrix(preds, labels, out_path="outputs/confusion_matrix.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",       default="models/woundscope_v3.pth")
    parser.add_argument("--data_csv",   default="dataset/labels.csv")
    parser.add_argument("--img_root",   default="dataset/wound_images")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)

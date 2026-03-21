"""
WoundScope utilities: metrics, checkpoint I/O, Grad-CAM helpers.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_auc_score, cohen_kappa_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from data_loader import WOUND_CLASSES, SEVERITY_UNKNOWN


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model_fn, loader, device):
    """
    Run inference over loader.

    model_fn(imgs, locs) may return:
      - a single tensor  → wound-type logits only
      - a 2-tuple        → (wound_logits, severity_logits)

    Returns:
      acc, macro_f1, wound_preds (ndarray), wound_labels (ndarray), sev_acc (float or None)
    """
    all_wound_preds,  all_wound_labels  = [], []
    all_wound_probs                     = []
    all_sev_preds,    all_sev_labels    = [], []

    with torch.no_grad():
        for batch in loader:
            imgs, locs, labels, severity = batch
            imgs, locs = imgs.to(device), locs.to(device)

            out = model_fn(imgs, locs)
            if isinstance(out, tuple):
                wound_logits, sev_logits = out
            else:
                wound_logits = out
                sev_logits   = None

            wound_probs = F.softmax(wound_logits, dim=1).cpu().numpy()
            wound_preds = wound_probs.argmax(axis=1)

            all_wound_preds.extend(wound_preds.tolist())
            all_wound_labels.extend(labels.numpy().tolist())
            all_wound_probs.append(wound_probs)

            if sev_logits is not None:
                sev_preds = sev_logits.argmax(dim=1).cpu().numpy()
                sev_true  = severity.numpy()
                sev_mask  = sev_true >= 0
                if sev_mask.any():
                    all_sev_preds.extend(sev_preds[sev_mask].tolist())
                    all_sev_labels.extend(sev_true[sev_mask].tolist())

    wound_preds_arr  = np.array(all_wound_preds)
    wound_labels_arr = np.array(all_wound_labels)
    wound_probs_arr  = np.vstack(all_wound_probs)

    acc = accuracy_score(wound_labels_arr, wound_preds_arr)
    f1  = f1_score(wound_labels_arr, wound_preds_arr, average="macro", zero_division=0)

    sev_acc = (
        accuracy_score(all_sev_labels, all_sev_preds)
        if all_sev_labels else None
    )

    return acc, f1, wound_preds_arr, wound_labels_arr, sev_acc


def per_class_auc(labels, probs, class_names=None):
    """
    Compute one-vs-rest AUC-ROC for each class.
    Returns dict {class_name: auc} and macro-average AUC.
    """
    class_names = class_names or WOUND_CLASSES
    n_classes   = probs.shape[1]
    aucs        = {}

    for i, name in enumerate(class_names[:n_classes]):
        binary_labels = (labels == i).astype(int)
        if binary_labels.sum() == 0 or (1 - binary_labels).sum() == 0:
            aucs[name] = float("nan")
            continue
        try:
            aucs[name] = roc_auc_score(binary_labels, probs[:, i])
        except ValueError:
            aucs[name] = float("nan")

    valid = [v for v in aucs.values() if not np.isnan(v)]
    macro_auc = float(np.mean(valid)) if valid else float("nan")
    return aucs, macro_auc


def severity_metrics(sev_preds, sev_labels):
    """
    Metrics for severity grading on labelled samples.
    Returns dict with accuracy, weighted Cohen's kappa, and MAE.
    """
    if len(sev_labels) == 0:
        return {}
    acc   = accuracy_score(sev_labels, sev_preds)
    kappa = cohen_kappa_score(sev_labels, sev_preds, weights="quadratic")
    mae   = float(np.mean(np.abs(np.array(sev_preds) - np.array(sev_labels))))
    return {"sev_accuracy": acc, "weighted_kappa": kappa, "sev_mae": mae}


def print_report(preds, labels, probs=None, sev_preds=None, sev_labels=None,
                 class_names=None, out_path=None):
    class_names = class_names or WOUND_CLASSES
    n = len(class_names)

    report = classification_report(
        labels, preds,
        labels=list(range(n)),
        target_names=class_names,
        zero_division=0,
    )
    cm_arr = confusion_matrix(labels, preds, labels=list(range(n)))

    lines = [
        "Classification Report:",
        report,
        "Confusion Matrix (rows=true, cols=pred):",
        "  " + "  ".join(f"{c[:8]:>8}" for c in class_names),
    ]
    for i, row in enumerate(cm_arr):
        lines.append(f"  {class_names[i][:8]:>8}  " + "  ".join(f"{v:8d}" for v in row))

    if probs is not None:
        auc_dict, macro_auc = per_class_auc(labels, probs, class_names)
        lines.append(f"\nAUC-ROC (macro avg): {macro_auc:.4f}")
        for name, auc in auc_dict.items():
            lines.append(f"  {name:<14} {auc:.4f}" if not np.isnan(auc) else f"  {name:<14}  N/A")

    if sev_preds is not None and sev_labels is not None and len(sev_labels) > 0:
        sev = severity_metrics(sev_preds, sev_labels)
        lines.append(
            f"\nSeverity — acc: {sev['sev_accuracy']:.4f}  "
            f"kappa: {sev['weighted_kappa']:.4f}  "
            f"MAE: {sev['sev_mae']:.4f}  "
            f"(n={len(sev_labels)})"
        )

    text = "\n".join(lines)
    print(text)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
    return text


def plot_confusion_matrix(preds, labels, class_names=None, out_path="outputs/confusion_matrix.png"):
    """Save a colour-coded confusion matrix as a PNG."""
    class_names = class_names or WOUND_CLASSES
    n           = len(class_names)
    cm_arr      = confusion_matrix(labels, preds, labels=list(range(n)))

    # Normalise rows to [0, 1] for colour, keep raw counts as text
    row_sums = cm_arr.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm  = cm_arr.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(n), yticks=range(n),
        xticklabels=class_names, yticklabels=class_names,
        ylabel="True label", xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

    thresh = 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    color=color, fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {out_path}")


# ── Checkpointing ──────────────────────────────────────────────────────────────

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device)


# ── Grad-CAM ───────────────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM for CNN models (ResNet etc.) — hooks on a target conv layer."""

    def __init__(self, model, target_layer):
        self.model       = model
        self.target_layer = target_layer
        self.gradients   = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, img_tensor, class_idx=None):
        self.model.eval()
        logits = self.model(img_tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1).squeeze()
        cam     = F.relu(cam).cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


class ViTGradCAM:
    """Grad-CAM for ViT models (timm). Hooks on the last transformer block."""

    def __init__(self, model):
        self.model       = model   # expects WoundScope with .backbone (ViT, num_classes=0)
        self.activations = None
        self.gradients   = None
        last_block = model.backbone.blocks[-1]
        last_block.register_forward_hook(self._forward_hook)
        last_block.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output[:, 1:, :].detach()   # skip CLS token

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0][:, 1:, :].detach()

    def generate(self, img_tensor, loc_tensor, class_idx=None):
        self.model.eval()
        wound_logits, _ = self.model(img_tensor, loc_tensor)
        if class_idx is None:
            class_idx = int(wound_logits.argmax(dim=1).item())

        self.model.zero_grad()
        wound_logits[0, class_idx].backward()

        cam = (self.gradients * self.activations).sum(dim=2)
        cam = F.relu(cam).squeeze(0)

        h = w = int(cam.shape[0] ** 0.5)
        cam = cam.reshape(h, w).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def overlay_gradcam(pil_image, cam, alpha=0.5):
    img_np = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0
    heatmap = cm.jet(cam)[..., :3]
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224))
    ).astype(np.float32) / 255.0

    overlay = alpha * heatmap_resized + (1 - alpha) * img_np
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


# ── Training helpers ────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("No GPU found, using CPU. Training will be slower.")
    return device


def plot_training_curves(train_losses, val_losses, val_accs, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses,   label="Val Loss")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(); ax1.set_title("Loss Curves")

    ax2.plot(val_accs, label="Val Accuracy", color="green")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
    ax2.legend(); ax2.set_title("Validation Accuracy")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Training curves saved: {out_path}")

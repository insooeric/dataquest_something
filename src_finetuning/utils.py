"""
WoundScope utilities: metrics, checkpoint I/O, Grad-CAM, training helpers.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_auc_score, cohen_kappa_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from data_loader import WOUND_CLASSES, SEVERITY_UNKNOWN


# ── Evaluation ──────────────────────────────────────────────────────────────────

def evaluate(model_fn, loader, device):
    all_wound_preds, all_wound_labels = [], []
    all_wound_probs                   = []
    all_sev_preds,   all_sev_labels   = [], []

    with torch.no_grad():
        for imgs, locs, labels, severity in loader:
            imgs, locs = imgs.to(device), locs.to(device)
            out = model_fn(imgs, locs)
            wound_logits, sev_logits = out if isinstance(out, tuple) else (out, None)

            wound_probs = F.softmax(wound_logits, dim=1).cpu().numpy()
            all_wound_preds.extend(wound_probs.argmax(axis=1).tolist())
            all_wound_labels.extend(labels.numpy().tolist())
            all_wound_probs.append(wound_probs)

            if sev_logits is not None:
                sev_preds = sev_logits.argmax(dim=1).cpu().numpy()
                sev_true  = severity.numpy()
                sev_mask  = sev_true >= 0
                if sev_mask.any():
                    all_sev_preds.extend(sev_preds[sev_mask].tolist())
                    all_sev_labels.extend(sev_true[sev_mask].tolist())

    preds_arr  = np.array(all_wound_preds)
    labels_arr = np.array(all_wound_labels)
    probs_arr  = np.vstack(all_wound_probs)

    acc = accuracy_score(labels_arr, preds_arr)
    f1  = f1_score(labels_arr, preds_arr, average="macro", zero_division=0)
    sev_acc = accuracy_score(all_sev_labels, all_sev_preds) if all_sev_labels else None

    return acc, f1, preds_arr, labels_arr, sev_acc, probs_arr, all_sev_preds, all_sev_labels


def per_class_auc(labels, probs, class_names=None):
    class_names = class_names or WOUND_CLASSES
    aucs = {}
    for i, name in enumerate(class_names[:probs.shape[1]]):
        binary = (labels == i).astype(int)
        if binary.sum() == 0 or (1 - binary).sum() == 0:
            aucs[name] = float("nan")
            continue
        try:
            aucs[name] = roc_auc_score(binary, probs[:, i])
        except ValueError:
            aucs[name] = float("nan")
    valid = [v for v in aucs.values() if not np.isnan(v)]
    return aucs, float(np.mean(valid)) if valid else float("nan")


def severity_metrics(sev_preds, sev_labels):
    if not sev_labels:
        return {}
    acc = accuracy_score(sev_labels, sev_preds)
    try:
        kappa = cohen_kappa_score(sev_labels, sev_preds, weights="quadratic")
    except ValueError:
        kappa = float("nan")
    mae = float(np.mean(np.abs(np.array(sev_preds) - np.array(sev_labels))))
    return {"sev_accuracy": acc, "weighted_kappa": kappa, "sev_mae": mae}


def print_report(preds, labels, probs=None, sev_preds=None, sev_labels=None,
                 class_names=None, out_path=None):
    class_names = class_names or WOUND_CLASSES
    n = len(class_names)
    report = classification_report(labels, preds, labels=list(range(n)),
                                   target_names=class_names, zero_division=0)
    cm_arr = confusion_matrix(labels, preds, labels=list(range(n)))

    lines = [
        "Classification Report:", report,
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

    if sev_preds and sev_labels:
        sev = severity_metrics(sev_preds, sev_labels)
        lines.append(
            f"\nSeverity — acc: {sev['sev_accuracy']:.4f}  "
            f"kappa: {sev['weighted_kappa']:.4f}  MAE: {sev['sev_mae']:.4f}  (n={len(sev_labels)})"
        )

    text = "\n".join(lines)
    print(text)
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            f.write(text)
    return text


def plot_confusion_matrix(preds, labels, class_names=None, out_path="outputs/confusion_matrix.png"):
    class_names = class_names or WOUND_CLASSES
    n      = len(class_names)
    cm_arr = confusion_matrix(labels, preds, labels=list(range(n)))
    cm_norm = cm_arr.astype(float) / cm_arr.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=range(n), yticks=range(n), xticklabels=class_names,
           yticklabels=class_names, ylabel="True label", xlabel="Predicted label",
           title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {out_path}")


# ── Checkpointing ───────────────────────────────────────────────────────────────

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=False)


# ── Grad-CAM (CNN) ──────────────────────────────────────────────────────────────

class GradCAM:
    """Grad-CAM for CNN models — hooks on a target conv layer."""

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "activations", o.detach()))
        target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0].detach()))

    def generate(self, img_tensor, loc_tensor=None, class_idx=None):
        self.model.eval()
        out = self.model(img_tensor, loc_tensor) if loc_tensor is not None else self.model(img_tensor)
        logits = out[0] if isinstance(out, tuple) else out
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1).squeeze()).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def overlay_gradcam(pil_image, cam, alpha=0.5):
    img_np   = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0
    heatmap  = cm.jet(cam)[..., :3]
    heatmap  = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224))).astype(np.float32) / 255.0
    overlay  = np.clip((alpha * heatmap + (1 - alpha) * img_np) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


# ── Device / plots ──────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        d = torch.device("cpu")
        print("No GPU found, using CPU. Training will be slower.")
    return d


def plot_training_curves(train_losses, val_losses, val_accs, out_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses,   label="Val Loss")
    ax1.set(xlabel="Epoch", ylabel="Loss", title="Loss Curves"); ax1.legend()
    ax2.plot(val_accs, label="Val Accuracy", color="green")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Validation Accuracy"); ax2.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path); plt.close()
    print(f"Training curves saved: {out_path}")

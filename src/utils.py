"""
WoundScope utilities: metrics, checkpoint I/O, Grad-CAM helpers.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

from data_loader import WOUND_CLASSES


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model_fn, loader, device):
    """Run inference, return (accuracy, macro_f1, all_preds, all_labels)."""
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                imgs, locs, labels = batch
                imgs, locs = imgs.to(device), locs.to(device)
                logits = model_fn(imgs, locs)
            else:
                imgs, labels = batch
                imgs = imgs.to(device)
                logits = model_fn(imgs)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1, np.array(all_preds), np.array(all_labels)


def print_report(preds, labels, out_path=None):
    report = classification_report(labels, preds, target_names=WOUND_CLASSES)
    cm = confusion_matrix(labels, preds)
    text = f"Classification Report:\n{report}\nConfusion Matrix:\n{cm}"
    print(text)
    if out_path:
        with open(out_path, "w") as f:
            f.write(text)
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")


def load_checkpoint(path, device="cpu"):
    return torch.load(path, map_location=device)


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """Simple Grad-CAM for ResNet/EfficientNet — hooks on the last conv layer."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
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
        """
        img_tensor: (1, C, H, W) on model device
        Returns: heatmap as (H, W) numpy array, normalized 0-1
        """
        self.model.eval()
        logits = self.model(img_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, class_idx].backward()

        # Pool gradients over spatial dims
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1).squeeze()   # (H, W)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


def overlay_gradcam(pil_image, cam, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on original PIL image.
    Returns a PIL Image (RGB).
    """
    img_np = np.array(pil_image.resize((224, 224))).astype(np.float32) / 255.0
    heatmap = cm.jet(cam)[..., :3]  # (H, W, 3)
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224))
    ).astype(np.float32) / 255.0

    overlay = alpha * heatmap_resized + (1 - alpha) * img_np
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

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
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("Loss Curves")

    ax2.plot(val_accs, label="Val Accuracy", color="green")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.set_title("Validation Accuracy")

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Training curves saved: {out_path}")

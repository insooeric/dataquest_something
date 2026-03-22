"""
WoundScope Dataset Loader
Expanded: 7 wound types with optional severity grading.

Unified severity scale:
  -1 = unknown / not applicable
   0 = mild     (Stage I  / 1st-degree burn / superficial)
   1 = moderate (Stage II / 2nd-degree burn / partial thickness)
   2 = severe   (Stage III / 3rd-degree burn / full thickness)
   3 = critical (Stage IV / deep tissue injury)
"""

import os
import re
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import torch

# ── Wound taxonomy ─────────────────────────────────────────────────────────────
WOUND_CLASSES = [
    "Diabetic",    # 0 – diabetic ulcer
    "Pressure",    # 1 – pressure ulcer / bedsore
    "Surgical",    # 2 – post-operative wound
    "Venous",      # 3 – venous leg ulcer
    "Arterial",    # 4 – arterial / ischemic ulcer
    "Burns",       # 5 – burn wound (all degrees)
    "Laceration",  # 6 – laceration / traumatic wound
]

# ── Severity scale ─────────────────────────────────────────────────────────────
SEVERITY_UNKNOWN = -1   # no label available
NUM_SEVERITY     = 4    # valid ordinal values: 0, 1, 2, 3

# Human-readable severity labels per wound type (for UI / reports)
SEVERITY_NAMES_BY_TYPE = {
    "Pressure": ["Stage I", "Stage II", "Stage III", "Stage IV"],
    "Burns":    ["1st Degree", "2nd Degree", "3rd Degree", "Deep Burn"],
}

# ── Body locations (6 zones, HAM10000-aligned) ─────────────────────────────────
BODY_LOCATIONS = [
    "head_neck", "chest", "abdomen", "back",
    "upper_extremity", "lower_extremity",
]
NUM_LOCATIONS = len(BODY_LOCATIONS)

# ── Transforms ─────────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

VAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class WoundDataset(Dataset):
    """
    CSV columns: image_path, wound_type, location, location_idx, [severity]

    __getitem__ returns: (image, loc_idx, wound_label, severity_label)
      severity_label = -1 (SEVERITY_UNKNOWN) when not available
    """

    def __init__(self, df, img_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["image_path"].replace("\\", "/"))
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label    = WOUND_CLASSES.index(row["wound_type"])
        loc      = int(row["location_idx"]) if "location_idx" in row else 0
        severity = (
            int(row["severity"])
            if "severity" in row and pd.notna(row["severity"])
            else SEVERITY_UNKNOWN
        )

        return image, loc, label, severity


def build_dataloaders(csv_path, img_root, batch_size=32, seed=42):
    """Load CSV, split 70/15/15, return train/val/test DataLoaders."""
    df = pd.read_csv(csv_path)

    if "location_idx" not in df.columns:
        df["location_idx"] = df["location"].apply(
            lambda x: BODY_LOCATIONS.index(x) if x in BODY_LOCATIONS else 0
        )

    if "severity" not in df.columns:
        df["severity"] = SEVERITY_UNKNOWN

    df["_base"] = df["image_path"].apply(
        lambda p: re.sub(r"_\d+\.(jpg|jpeg|png)$", "", p.replace("\\", "/"), flags=re.IGNORECASE)
    )
    bases = df[["_base", "wound_type"]].drop_duplicates("_base")

    def _stratify(frame):
        counts = frame["wound_type"].value_counts()
        return frame["wound_type"] if (counts >= 2).all() else None

    train_bases, temp_bases = train_test_split(bases, test_size=0.30, stratify=_stratify(bases), random_state=seed)
    val_bases,   test_bases = train_test_split(temp_bases, test_size=0.50, stratify=_stratify(temp_bases), random_state=seed)

    train_df = df[df["_base"].isin(train_bases["_base"])].drop(columns="_base")
    val_df   = df[df["_base"].isin(val_bases["_base"])].drop(columns="_base")
    test_df  = df[df["_base"].isin(test_bases["_base"])].drop(columns="_base")

    train_ds = WoundDataset(train_df, img_root, TRAIN_TRANSFORM)
    val_ds   = WoundDataset(val_df,   img_root, VAL_TRANSFORM)
    test_ds  = WoundDataset(test_df,  img_root, VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def infer_labels_from_filenames(wound_images_dir):
    """
    Walk wound_images_dir, infer wound type from subfolder name.
    Returns a DataFrame with columns: image_path, wound_type, location, location_idx, severity.
    """
    records = []
    for root, _, files in os.walk(wound_images_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            rel_path = os.path.relpath(os.path.join(root, fname), wound_images_dir)
            folder   = os.path.basename(root)
            wound_type = None
            for wt in WOUND_CLASSES:
                if wt.lower() in folder.lower() or wt.lower() in fname.lower():
                    wound_type = wt
                    break
            if wound_type is None:
                continue
            records.append({
                "image_path":  rel_path,
                "wound_type":  wound_type,
                "location":    "lower_extremity",
                "location_idx": BODY_LOCATIONS.index("lower_extremity"),
                "severity":    SEVERITY_UNKNOWN,
            })
    return pd.DataFrame(records)

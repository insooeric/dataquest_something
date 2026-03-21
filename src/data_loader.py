"""
WoundScope Dataset Loader
AZH wound dataset: 4 classes - diabetic, pressure, surgical, venous
"""

import os
import json
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as T
import torch

# 4 wound types from AZH dataset
WOUND_CLASSES = ["Diabetic", "Pressure", "Surgical", "Venous"]

# Body locations (6 zones aligned with HAM10000 localization labels)
BODY_LOCATIONS = [
    "head_neck", "chest", "abdomen", "back",
    "upper_extremity", "lower_extremity"
]
NUM_LOCATIONS = len(BODY_LOCATIONS)

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
    Expects a CSV with columns: image_path, wound_type, location
    wound_type: one of WOUND_CLASSES
    location: one of BODY_LOCATIONS (or int index)
    """

    def __init__(self, df, img_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row["image_path"])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = WOUND_CLASSES.index(row["wound_type"])
        loc = int(row["location_idx"]) if "location_idx" in row else 0

        return image, loc, label


def build_dataloaders(csv_path, img_root, batch_size=32, seed=42):
    """Load CSV, split, return train/val/test DataLoaders."""
    df = pd.read_csv(csv_path)

    # Encode location string -> int if needed
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
    val_ds = WoundDataset(val_df, img_root, VAL_TRANSFORM)
    test_ds = WoundDataset(test_df, img_root, VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def infer_labels_from_filenames(wound_images_dir):
    """
    Auto-build a DataFrame by parsing filenames from the AZH dataset.
    AZH filenames typically contain the wound type in the folder structure:
      wound_images/Diabetic/img001.jpg  OR
      wound_images/img_diabetic_001.jpg
    Returns a DataFrame ready for WoundDataset.
    """
    records = []
    for root, dirs, files in os.walk(wound_images_dir):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            rel_path = os.path.relpath(os.path.join(root, fname), wound_images_dir)
            # Try to infer wound type from parent folder name
            folder = os.path.basename(root)
            wound_type = None
            for wt in WOUND_CLASSES:
                if wt.lower() in folder.lower() or wt.lower() in fname.lower():
                    wound_type = wt
                    break
            if wound_type is None:
                continue  # skip unrecognized
            records.append({
                "image_path": rel_path,
                "wound_type": wound_type,
                "location": "lower_extremity",  # default; update from metadata CSV if available
                "location_idx": BODY_LOCATIONS.index("lower_extremity"),
            })
    return pd.DataFrame(records)

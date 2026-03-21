"""
WoundScope – Download additional wound datasets from Kaggle and merge into labels.csv.

Setup (run once on server):
    export KAGGLE_USERNAME=<your_kaggle_username>
    export KAGGLE_KEY=KGAT_f269718d2e6e028d22e163aeaf753648

Usage:
    python3 src/fetch_extra_data.py \
        --wound_dir dataset/wound_images \
        --out_csv   dataset/labels.csv

What it does:
    1. Searches Kaggle for wound/ulcer image datasets
    2. Downloads and extracts relevant ones
    3. Maps their classes to our 4 types (Diabetic, Pressure, Surgical, Venous)
    4. Appends new rows to labels.csv (deduplicates by image_path)
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import WOUND_CLASSES, BODY_LOCATIONS
from prepare_dataset import assign_location

# ──────────────────────────────────────────────────────────────────────────────
# Kaggle datasets to try (slug: username/dataset-name)
# Add / remove as needed
# ──────────────────────────────────────────────────────────────────────────────
TARGET_DATASETS = [
    "ibrahimfateen/wound-classification",   # multi-type: diabetic/pressure/surgical/venous
    "laithjj/diabetic-foot-ulcer-dfu",      # DFU-specific, ~5500 images
    "sinemgokoz/pressure-ulcers-stages",    # pressure ulcer stages
]

# How to map folder/file keywords → our 4 classes
CLASS_MAP = {
    "diabetic":  "Diabetic",
    "diab":      "Diabetic",
    "dfu":       "Diabetic",   # diabetic foot ulcer
    "pressure":  "Pressure",
    "decubitus": "Pressure",
    "bedsore":   "Pressure",
    "surgical":  "Surgical",
    "postop":    "Surgical",
    "post_op":   "Surgical",
    "incision":  "Surgical",
    "venous":    "Venous",
    "vascular":  "Venous",
    "stasis":    "Venous",
}


def setup_kaggle_credentials():
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        print("ERROR: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        print("  export KAGGLE_USERNAME=<your_username>")
        print("  export KAGGLE_KEY=KGAT_f269718d2e6e028d22e163aeaf753648")
        sys.exit(1)

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    creds_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(creds_path, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(creds_path, 0o600)
    print(f"Kaggle credentials written to {creds_path}")


def download_dataset(slug, dest_dir):
    """Download and extract a Kaggle dataset. Returns extracted path or None."""
    name = slug.split("/")[-1]
    out_path = os.path.join(dest_dir, name)
    if os.path.exists(out_path):
        print(f"  Already exists: {out_path}")
        return out_path

    print(f"  Downloading {slug} ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", dest_dir, "--unzip"],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        print("  ERROR: kaggle CLI not found. Run: pip install kaggle")
        return None

    # If unzip didn't create a subdir, make one
    zip_path = os.path.join(dest_dir, f"{name}.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_path)
        os.remove(zip_path)

    return out_path if os.path.exists(out_path) else dest_dir


def infer_class(path_str):
    """Guess wound type from file/folder name."""
    p = path_str.lower()
    for keyword, cls in CLASS_MAP.items():
        if keyword in p:
            return cls
    return None


def scan_directory(root, img_root):
    """Walk root, infer class from path, return list of dicts."""
    records = []
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            full = os.path.join(dirpath, fname)
            # Try to infer class from folder or filename
            rel_from_root = os.path.relpath(full, root)
            wound_type = infer_class(dirpath) or infer_class(fname)
            if wound_type is None:
                continue

            # Copy into our wound_images dir under the right class folder
            dest_folder = os.path.join(img_root, wound_type)
            os.makedirs(dest_folder, exist_ok=True)
            dest_file = os.path.join(dest_folder, f"ext_{abs(hash(full))}.jpg")
            if not os.path.exists(dest_file):
                shutil.copy2(full, dest_file)

            rel_path = os.path.relpath(dest_file, img_root)
            location = assign_location(wound_type, seed=hash(full))
            records.append({
                "image_path":   rel_path,
                "wound_type":   wound_type,
                "location":     location,
                "location_idx": BODY_LOCATIONS.index(location),
            })
    return records


def main(args):
    setup_kaggle_credentials()

    tmp_dir = "dataset/_kaggle_tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(args.wound_dir, exist_ok=True)

    all_new_records = []
    for slug in TARGET_DATASETS:
        print(f"\n── {slug}")
        extracted = download_dataset(slug, tmp_dir)
        if extracted is None:
            continue
        records = scan_directory(extracted, args.wound_dir)
        print(f"  Found {len(records)} usable images")
        all_new_records.extend(records)

    if not all_new_records:
        print("\nNo new images found. Check dataset slugs or credentials.")
        return

    new_df = pd.DataFrame(all_new_records)

    # Merge with existing CSV (deduplicate by image_path)
    if os.path.exists(args.out_csv):
        existing = pd.read_csv(args.out_csv)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="image_path")
    else:
        combined = new_df

    combined.to_csv(args.out_csv, index=False)

    print(f"\n── Dataset summary after merge:")
    print(combined["wound_type"].value_counts())
    print(f"\nTotal: {len(combined)} images → {args.out_csv}")

    # Clean up temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wound_dir", default="dataset/wound_images")
    parser.add_argument("--out_csv",   default="dataset/labels.csv")
    args = parser.parse_args()
    main(args)

"""
WoundScope – Download additional wound datasets and merge into labels.csv.

Adds wound types: Arterial, Burns, Laceration
Adds severity labels for: Pressure (NPUAP Stage I–IV), Burns (1st–3rd degree)

─── Data sources ──────────────────────────────────────────────────────────────

1. Kaggle (requires KAGGLE_USERNAME + KAGGLE_KEY env vars):
   - ibrahimfateen/wound-classification     (multi-type)
   - laithjj/diabetic-foot-ulcer-dfu        (DFU, ~5 500 images)
   - sinemgokoz/pressure-ulcers-stages      (NPUAP Stage I–IV)
   - leoscode/wound-segmentation-images     (2 760 mixed images)
   - yasinpratomo/wound-dataset             (general)

2. Roboflow burns (requires ROBOFLOW_API_KEY env var):
   - binussss/burn-wound-classification     (~3 700 images, degree labels)

3. Medetec (manual):
   - Download from http://medetec.co.uk/files/medetec-image-databases.html
   - Point --medetec_dir at the extracted folder

─── Setup ─────────────────────────────────────────────────────────────────────

    export KAGGLE_USERNAME=<your_username>
    export KAGGLE_KEY=<your_key>
    export ROBOFLOW_API_KEY=<your_key>   # optional

─── Usage ─────────────────────────────────────────────────────────────────────

    python src/fetch_extra_data.py \\
        --wound_dir  dataset/wound_images \\
        --out_csv    dataset/labels.csv \\
        --medetec_dir /path/to/medetec   # optional
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import WOUND_CLASSES, BODY_LOCATIONS, SEVERITY_UNKNOWN
from prepare_dataset import assign_location


# ── Kaggle dataset slugs ────────────────────────────────────────────────────────
KAGGLE_DATASETS = [
    "ibrahimfateen/wound-classification",
    "laithjj/diabetic-foot-ulcer-dfu",
    "sinemgokoz/pressure-ulcers-stages",   # NPUAP staged
    "leoscode/wound-segmentation-images",  # large mixed dataset
    "yasinpratomo/wound-dataset",
]

# ── Keyword → wound type ────────────────────────────────────────────────────────
CLASS_MAP = {
    "diabetic":   "Diabetic",
    "diab":       "Diabetic",
    "dfu":        "Diabetic",
    "pressure":   "Pressure",
    "decubitus":  "Pressure",
    "bedsore":    "Pressure",
    "surgical":   "Surgical",
    "postop":     "Surgical",
    "post_op":    "Surgical",
    "incision":   "Surgical",
    "venous":     "Venous",
    "vascular":   "Venous",
    "stasis":     "Venous",
    "arterial":   "Arterial",
    "ischemic":   "Arterial",
    "ischaemic":  "Arterial",
    "burn":       "Burns",
    "scald":      "Burns",
    "thermal":    "Burns",
    "laceration": "Laceration",
    "lacerat":    "Laceration",
    "abrasion":   "Laceration",
    "traumatic":  "Laceration",
}

# ── Keyword → unified severity (0=mild … 3=critical) ──────────────────────────
SEVERITY_MAP = {
    # Pressure ulcer NPUAP staging
    "stage1": 0, "stage_1": 0, "stagei": 0, "stage_i": 0,
    "stage2": 1, "stage_2": 1, "stageii": 1, "stage_ii": 1,
    "stage3": 2, "stage_3": 2, "stageiii": 2, "stage_iii": 2,
    "stage4": 3, "stage_4": 3, "stageiv": 3, "stage_iv": 3,
    "deep_tissue": 3, "dti": 3, "unstageable": 3,
    # Burn degree
    "first_degree": 0, "1st_degree": 0, "degree_1": 0, "superficial": 0,
    "second_degree": 1, "2nd_degree": 1, "degree_2": 1, "partial_thickness": 1,
    "third_degree": 2, "3rd_degree": 2, "degree_3": 2, "full_thickness": 2,
    # Generic ordinal fallbacks
    "mild": 0, "moderate": 1, "severe": 2, "critical": 3,
    "grade1": 0, "grade2": 1, "grade3": 2, "grade4": 3,
}


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _normalise(s):
    """Lowercase and collapse separators."""
    return re.sub(r"[\s\-]+", "_", s.lower())


def infer_class(path_str):
    p = _normalise(path_str)
    for keyword, cls in CLASS_MAP.items():
        if keyword in p:
            return cls
    return None


def infer_severity(path_str):
    p = _normalise(path_str)
    for keyword, sev in SEVERITY_MAP.items():
        if keyword in p:
            return sev
    return SEVERITY_UNKNOWN


def setup_kaggle_credentials():
    username = os.environ.get("KAGGLE_USERNAME")
    key      = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        print("  SKIP: Set KAGGLE_USERNAME and KAGGLE_KEY to enable Kaggle downloads.")
        return False
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    creds_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(creds_path, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(creds_path, 0o600)
    print(f"  Kaggle credentials written → {creds_path}")
    return True


def download_kaggle_dataset(slug, dest_dir):
    """Download + unzip a Kaggle dataset. Returns extracted path or None."""
    name     = slug.split("/")[-1]
    out_path = os.path.join(dest_dir, name)
    if os.path.exists(out_path):
        print(f"  Already cached: {out_path}")
        return out_path
    print(f"  Downloading {slug} …")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug, "-p", dest_dir, "--unzip"],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {e.stderr.strip()}")
        return None
    except FileNotFoundError:
        print("  ERROR: kaggle CLI not found — run: pip install kaggle")
        return None
    zip_path = os.path.join(dest_dir, f"{name}.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_path)
        os.remove(zip_path)
    return out_path if os.path.exists(out_path) else dest_dir


def download_roboflow_burns(dest_dir):
    """
    Download burn wound dataset from Roboflow using the roboflow SDK.
    Requires: pip install roboflow  and  ROBOFLOW_API_KEY env var.
    Returns extracted path or None.
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("  SKIP: Set ROBOFLOW_API_KEY to download Roboflow burn dataset.")
        return None
    try:
        from roboflow import Roboflow
    except ImportError:
        print("  SKIP: roboflow package not installed — run: pip install roboflow")
        return None

    out_path = os.path.join(dest_dir, "burn-wound-classification")
    if os.path.exists(out_path):
        print(f"  Already cached: {out_path}")
        return out_path

    print("  Downloading binussss/burn-wound-classification from Roboflow …")
    try:
        rf      = Roboflow(api_key=api_key)
        project = rf.workspace("binussss").project("burn-wound-classification")
        dataset = project.version(1).download("folder", location=out_path)
        return out_path
    except Exception as e:
        print(f"  FAILED: {e}")
        return None


def scan_directory(root, img_root):
    """
    Walk root, infer wound type + severity from path, copy images into img_root.
    Returns list of record dicts (image_path, wound_type, location, location_idx, severity).
    """
    records = []
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            full        = os.path.join(dirpath, fname)
            rel_in_src  = os.path.relpath(full, root)

            # Infer class from folder path, then filename
            wound_type = infer_class(dirpath) or infer_class(fname)
            if wound_type is None:
                continue

            # Infer severity from folder path, then filename
            severity = infer_severity(dirpath)
            if severity == SEVERITY_UNKNOWN:
                severity = infer_severity(fname)

            # Copy into wound_images/<WoundType>/
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
                "severity":     severity,
            })
    return records


# ── Main ────────────────────────────────────────────────────────────────────────

def main(args):
    tmp_dir = "dataset/_extra_tmp"
    os.makedirs(tmp_dir,       exist_ok=True)
    os.makedirs(args.wound_dir, exist_ok=True)

    all_new_records = []

    # ── 1. Kaggle ──────────────────────────────────────────────────────────────
    print("\n── Kaggle datasets")
    if setup_kaggle_credentials():
        for slug in KAGGLE_DATASETS:
            print(f"\n  {slug}")
            extracted = download_kaggle_dataset(slug, tmp_dir)
            if extracted is None:
                continue
            records = scan_directory(extracted, args.wound_dir)
            print(f"  → {len(records)} usable images")
            all_new_records.extend(records)
    else:
        print("  Skipping Kaggle (no credentials).")

    # ── 2. Roboflow burns ──────────────────────────────────────────────────────
    print("\n── Roboflow: burn-wound-classification")
    extracted = download_roboflow_burns(tmp_dir)
    if extracted:
        records = scan_directory(extracted, args.wound_dir)
        print(f"  → {len(records)} usable images")
        all_new_records.extend(records)

    # ── 3. Medetec (manual) ────────────────────────────────────────────────────
    if args.medetec_dir:
        print(f"\n── Medetec: {args.medetec_dir}")
        if os.path.isdir(args.medetec_dir):
            records = scan_directory(args.medetec_dir, args.wound_dir)
            print(f"  → {len(records)} usable images")
            all_new_records.extend(records)
        else:
            print("  WARNING: medetec_dir does not exist — skipping.")
    else:
        print("\n── Medetec: skipped (pass --medetec_dir to include)")

    # ── Merge with existing CSV ────────────────────────────────────────────────
    if not all_new_records:
        print("\nNo new images found. Check credentials or dataset paths.")
        return

    new_df = pd.DataFrame(all_new_records)

    if os.path.exists(args.out_csv):
        existing = pd.read_csv(args.out_csv)
        if "severity" not in existing.columns:
            existing["severity"] = SEVERITY_UNKNOWN
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset="image_path")
    else:
        combined = new_df

    combined.to_csv(args.out_csv, index=False)

    print(f"\n── Merged dataset summary:")
    print(combined["wound_type"].value_counts())
    print(f"\nSeverity distribution (−1 = unknown):")
    print(combined["severity"].value_counts().sort_index())
    print(f"\nTotal: {len(combined)} images → {args.out_csv}")

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wound_dir",   default="dataset/wound_images")
    parser.add_argument("--out_csv",     default="dataset/labels.csv")
    parser.add_argument("--medetec_dir", default=None,
                        help="Path to manually downloaded Medetec dataset folder")
    args = parser.parse_args()
    main(args)

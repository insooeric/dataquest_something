"""
WoundScope – Download ALL wound datasets and merge into labels.csv.

Run once to get everything:

    export KAGGLE_USERNAME=<your_username>
    export KAGGLE_KEY=<your_key>
    python src/fetch_extra_data.py

Optional: wipe and rebuild from scratch:
    rm -rf dataset/wound_images dataset/labels.csv
    python src/fetch_extra_data.py
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import WOUND_CLASSES, BODY_LOCATIONS, SEVERITY_UNKNOWN
from prepare_dataset import assign_location


# ── Kaggle dataset slugs (all downloaded every run) ─────────────────────────────
KAGGLE_DATASETS = [
    # Multi-type wound classification
    "ibrahimfateen/wound-classification",
    # Diabetic foot ulcer (~5 500 images)
    "laithjj/diabetic-foot-ulcer-dfu",
    # Pressure ulcers with NPUAP Stage I–IV labels
    "sinemgokoz/pressure-ulcers-stages",
    # Large mixed wound dataset
    "leoscode/wound-segmentation-images",
    # General wound dataset
    "yasinpratomo/wound-dataset",
    # Burns with degree labels
    "rashik/skin-burn-image-dataset",
    # Additional burn wound images
    "devzohaib/skin-burn-classification",
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
    "laseration": "Laceration",
    "lacerat":    "Laceration",
    "abrasion":   "Laceration",
    "traumatic":  "Laceration",
    "cut":        "Laceration",
}

# ── Keyword → unified severity (0=mild … 3=critical) ──────────────────────────
SEVERITY_MAP = {
    # Pressure ulcer NPUAP staging
    "stage1": 0, "stage_1": 0, "stagei": 0, "stage_i": 0,
    "stage2": 1, "stage_2": 1, "stageii": 1, "stage_ii": 1,
    "stage3": 2, "stage_3": 2, "stageiii": 2, "stage_iii": 2,
    "stage4": 3, "stage_4": 3, "stageiv": 3, "stage_iv": 3,
    "deep_tissue": 3, "dti": 3, "unstageable": 3,
    # Bare number folder names
    "1": 0, "2": 1, "3": 2, "4": 3,
    # Burn degree
    "first_degree": 0, "1st_degree": 0, "degree_1": 0, "superficial": 0,
    "second_degree": 1, "2nd_degree": 1, "degree_2": 1, "partial_thickness": 1,
    "third_degree": 2, "3rd_degree": 2, "degree_3": 2, "full_thickness": 2,
    # Generic ordinal fallbacks
    "mild": 0, "moderate": 1, "severe": 2, "critical": 3,
    "grade1": 0, "grade2": 1, "grade3": 2, "grade4": 3,
}


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _content_hash(path):
    """MD5 of file contents — same image from different sources gets same hash."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _normalise(s):
    return re.sub(r"[\s\-]+", "_", s.lower())


def infer_class(path_str):
    p = _normalise(path_str)
    for keyword, cls in CLASS_MAP.items():
        if keyword in p:
            return cls
    return None


def infer_severity(folder_name):
    p = _normalise(os.path.basename(folder_name))
    if p in SEVERITY_MAP:
        return SEVERITY_MAP[p]
    for keyword, sev in SEVERITY_MAP.items():
        if len(keyword) > 1 and keyword in p:
            return sev
    return SEVERITY_UNKNOWN


def setup_kaggle_credentials():
    username = os.environ.get("KAGGLE_USERNAME")
    key      = os.environ.get("KAGGLE_KEY")
    if not username or not key:
        print("ERROR: KAGGLE_USERNAME and KAGGLE_KEY must be set.")
        print("  export KAGGLE_USERNAME=<your_username>")
        print("  export KAGGLE_KEY=<your_key>")
        sys.exit(1)
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    creds_path = os.path.join(kaggle_dir, "kaggle.json")
    with open(creds_path, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(creds_path, 0o600)
    print(f"Kaggle credentials written → {creds_path}")


def download_kaggle_dataset(slug, dest_dir):
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
        sys.exit(1)
    zip_path = os.path.join(dest_dir, f"{name}.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(out_path)
        os.remove(zip_path)
    return out_path if os.path.exists(out_path) else dest_dir


def scan_directory(root, img_root, debug=False):
    if debug:
        folder_names = set()
        for dirpath, _, files in os.walk(root):
            if any(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in files):
                folder_names.add(os.path.basename(dirpath))
        print(f"  [debug] leaf folder names: {sorted(folder_names)[:30]}")

    records = []
    for dirpath, _, files in os.walk(root):
        for fname in files:
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            full       = os.path.join(dirpath, fname)
            wound_type = infer_class(dirpath) or infer_class(fname)
            if wound_type is None:
                continue

            severity = infer_severity(dirpath)
            if severity == SEVERITY_UNKNOWN:
                severity = infer_severity(fname)

            dest_folder = os.path.join(img_root, wound_type)
            os.makedirs(dest_folder, exist_ok=True)
            dest_file = os.path.join(dest_folder, f"ext_{_content_hash(full)}.jpg")
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
    setup_kaggle_credentials()

    tmp_dir = "dataset/_extra_tmp"
    os.makedirs(tmp_dir,        exist_ok=True)
    os.makedirs(args.wound_dir, exist_ok=True)

    all_new_records = []

    for slug in KAGGLE_DATASETS:
        print(f"\n── {slug}")
        extracted = download_kaggle_dataset(slug, tmp_dir)
        if extracted is None:
            print("  Skipping (download failed).")
            continue
        records  = scan_directory(extracted, args.wound_dir, debug=args.debug)
        with_sev = sum(1 for r in records if r["severity"] != SEVERITY_UNKNOWN)
        print(f"  → {len(records)} images  ({with_sev} with severity labels)")
        all_new_records.extend(records)

    # Optional: Medetec (manual download only)
    if args.medetec_dir:
        print(f"\n── Medetec: {args.medetec_dir}")
        if os.path.isdir(args.medetec_dir):
            records  = scan_directory(args.medetec_dir, args.wound_dir, debug=args.debug)
            with_sev = sum(1 for r in records if r["severity"] != SEVERITY_UNKNOWN)
            print(f"  → {len(records)} images  ({with_sev} with severity labels)")
            all_new_records.extend(records)
        else:
            print("  WARNING: path does not exist — skipping.")

    if not all_new_records:
        print("\nNo images found.")
        shutil.rmtree(tmp_dir, ignore_errors=True)
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

    print(f"\n── Dataset summary:")
    print(combined["wound_type"].value_counts())
    print(f"\nSeverity distribution (−1 = unknown):")
    print(combined["severity"].value_counts().sort_index())
    print(f"\nTotal: {len(combined)} images → {args.out_csv}")

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wound_dir",   default="dataset/wound_images")
    parser.add_argument("--out_csv",     default="dataset/labels.csv")
    parser.add_argument("--medetec_dir", default=None)
    parser.add_argument("--debug",       action="store_true")
    args = parser.parse_args()
    main(args)

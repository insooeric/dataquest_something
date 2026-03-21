"""
WoundScope – Dataset preparation script.

After downloading wound images, run this to generate dataset/labels.csv.

Usage:
    python src/prepare_dataset.py --wound_dir dataset/wound_images \
                                   --out_csv dataset/labels.csv
"""

import argparse
import os
import sys
import pandas as pd
import random

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import WOUND_CLASSES, BODY_LOCATIONS, SEVERITY_UNKNOWN, infer_labels_from_filenames


# Clinical location priors for each wound type
LOCATION_PRIORS = {
    "Diabetic":   ["lower_extremity"] * 8 + ["upper_extremity"] * 2,
    "Pressure":   ["back"] * 5 + ["lower_extremity"] * 2 + ["chest"] * 2 + ["head_neck"],
    "Surgical":   ["abdomen"] * 4 + ["chest"] * 4 + ["back"] * 2,
    "Venous":     ["lower_extremity"] * 9 + ["upper_extremity"],
    "Arterial":   ["lower_extremity"] * 8 + ["upper_extremity"] * 2,
    "Burns":      ["upper_extremity"] * 3 + ["lower_extremity"] * 3 + ["chest"] * 2 + ["back"] * 2,
    "Laceration": ["upper_extremity"] * 4 + ["lower_extremity"] * 3 + ["head_neck"] * 2 + ["abdomen"],
}


def assign_location(wound_type, seed=None):
    """Sample a plausible body location for a wound type."""
    if seed is not None:
        random.seed(seed)
    options = LOCATION_PRIORS.get(wound_type, BODY_LOCATIONS)
    return random.choice(options)


def main(args):
    print(f"Scanning: {args.wound_dir}")
    df = infer_labels_from_filenames(args.wound_dir)

    if df.empty:
        print("ERROR: No images found. Check that wound_dir has subdirs named after wound types.")
        sys.exit(1)

    if "severity" not in df.columns:
        df["severity"] = SEVERITY_UNKNOWN

    # Assign locations from clinical priors if all are still at the default
    if (df["location"] == "lower_extremity").all():
        print("Assigning locations from clinical priors (no metadata found).")
        df["location"] = [
            assign_location(wt, seed=i)
            for i, wt in enumerate(df["wound_type"])
        ]
        df["location_idx"] = df["location"].apply(BODY_LOCATIONS.index)

    print(f"\nDataset summary:")
    print(df["wound_type"].value_counts())
    print(f"\nLocation distribution:")
    print(df["location"].value_counts())
    print(f"\nSeverity distribution (−1 = unknown):")
    print(df["severity"].value_counts().sort_index())
    print(f"\nTotal: {len(df)} images")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wound_dir", default="dataset/wound_images")
    parser.add_argument("--out_csv",   default="dataset/labels.csv")
    args = parser.parse_args()
    main(args)

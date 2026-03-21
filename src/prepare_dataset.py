"""
WoundScope – Dataset preparation script.

After cloning the AZH repo, run this to generate dataset/labels.csv.

Usage:
    python src/prepare_dataset.py --wound_dir dataset/wound_images \
                                   --out_csv dataset/labels.csv

The AZH repo structure is typically:
    wound_images/
        Diabetic/  (or diabetic/)
        Pressure/
        Surgical/
        Venous/
"""

import argparse
import os
import sys
import pandas as pd
import random

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import WOUND_CLASSES, BODY_LOCATIONS, infer_labels_from_filenames


# AZH metadata: clinical priors mapped to the 6-zone scheme
LOCATION_PRIORS = {
    "Diabetic":  ["lower_extremity"] * 8 + ["upper_extremity"] * 2,
    "Pressure":  ["back"] * 5 + ["lower_extremity"] * 2 + ["chest"] * 2 + ["head_neck"],
    "Surgical":  ["abdomen"] * 4 + ["chest"] * 4 + ["back"] * 2,
    "Venous":    ["lower_extremity"] * 9 + ["upper_extremity"],
}


def assign_location(wound_type, seed=None):
    """Sample a plausible location for a wound type (for datasets without metadata)."""
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

    # If all locations are default, assign using priors
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
    print(f"\nTotal: {len(df)} images")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved: {args.out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wound_dir", default="dataset/wound_images")
    parser.add_argument("--out_csv", default="dataset/labels.csv")
    args = parser.parse_args()
    main(args)

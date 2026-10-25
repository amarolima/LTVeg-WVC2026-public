"""
Compares regenerated images against the fingerprints of the original dataset
(data/labels.csv) and, optionally, arranges them in the train/val/test folder
structure used by every script in this repository.

Each image is matched to the original by FILE NAME and classified as:

    IDENTICAL_FILE    same PNG bytes (sha256_file)
    IDENTICAL_PIXELS  same RGB pixels, different encoding (sha256_pixels)
    SAME_SCENE        pHash <= 10 and dHash <= 10: same place, small rendering,
                      compression or brightness differences
    CHECK             pHash or dHash between 11 and 17: possibly the same place with
                      a shift of several pixels or a newer acquisition; inspect it
    DIFFERENT         anything else: most likely a newer Google acquisition, or a
                      wrong tower
    MISSING           no file with this name in the input folder

The thresholds were calibrated on the 351 original images (docs/02_REGENERATING_THE_DATASET.md):
distinct images are >= 14 apart (1st percentile 22) except one pair of towers 2 m apart,
whereas the same image re-encoded, shifted by up to 5 px or brightened by 10 % stays <= 12.

Usage (from the repository root):
    python src/dataset/verify_dataset.py --images <folder with the PNGs>
    python src/dataset/verify_dataset.py --images <folder> --organize data/dataset
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprints import fingerprint, hamming

LABELS = Path("data/labels.csv")
SAME_SCENE_MAX = 10
CHECK_MAX = 17
EXACT = ("IDENTICAL_FILE", "IDENTICAL_PIXELS")


def classify(ref, fp):
    if fp is None:
        return "MISSING", None, None
    dp, dd = hamming(ref.phash, fp["phash"]), hamming(ref.dhash, fp["dhash"])
    if fp["sha256_file"] == ref.sha256_file:
        return "IDENTICAL_FILE", dp, dd
    if fp["sha256_pixels"] == ref.sha256_pixels:
        return "IDENTICAL_PIXELS", dp, dd
    if dp <= SAME_SCENE_MAX and dd <= SAME_SCENE_MAX:
        return "SAME_SCENE", dp, dd
    if min(dp, dd) <= CHECK_MAX:
        return "CHECK", dp, dd
    return "DIFFERENT", dp, dd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, type=Path, help="folder with the regenerated PNGs (searched recursively)")
    ap.add_argument("--organize", type=Path, help="copy the images into <dir>/{split}/{label}/ (e.g. data/dataset)")
    ap.add_argument("--accept", default="SAME_SCENE", choices=["IDENTICAL_FILE", "IDENTICAL_PIXELS", "SAME_SCENE", "CHECK", "ALL"],
                    help="weakest status copied by --organize (default: SAME_SCENE)")
    ap.add_argument("--report", type=Path, default=Path("results/dataset_verification.csv"))
    args = ap.parse_args()

    labels = pd.read_csv(LABELS)
    found = {p.name: p for p in args.images.rglob("*.png")}
    extra = sorted(set(found) - set(labels.file))

    rows = []
    for ref in labels.itertuples():
        p = found.get(ref.file)
        status, dp, dd = classify(ref, fingerprint(p) if p else None)
        rows.append(dict(file=ref.file, split=ref.split, label=ref.label, status=status,
                         phash_distance=dp, dhash_distance=dd, path=str(p) if p else ""))
    rep = pd.DataFrame(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(args.report, index=False)

    order = ["IDENTICAL_FILE", "IDENTICAL_PIXELS", "SAME_SCENE", "CHECK", "DIFFERENT", "MISSING"]
    counts = rep.status.value_counts().reindex(order, fill_value=0)
    print(f"{len(rep)} images expected, {len(found)} PNGs found in {args.images}")
    for s, n in counts.items():
        print(f"  {s:17s} {n:4d}")
    per = pd.crosstab([rep.split, rep.label], rep.status).reindex(columns=order, fill_value=0)
    print("  per split/label:\n" + per.to_string())
    if extra:
        print(f"  {len(extra)} PNGs not in labels.csv were ignored (e.g. {extra[:3]})")
    print(f"report: {args.report}")

    n_exact = counts[list(EXACT)].sum()
    if n_exact == len(rep):
        print("\nAll images are exact copies of the original dataset: every number of the paper is reproducible.")
    else:
        print(f"\nOnly {n_exact}/{len(rep)} images are exact copies. Retrained models and the colour baseline will"
              " not match the paper exactly; the notebook still reproduces every table from the stored results.")

    if args.organize:
        ok = order[:order.index(args.accept) + 1] if args.accept != "ALL" else order[:-1]
        sel = rep[rep.status.isin(ok)]
        for r in sel.itertuples():
            dst = args.organize / r.split / r.label / r.file
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(r.path, dst)
        print(f"\ncopied {len(sel)} images ({', '.join(ok)}) into {args.organize}/{{train,val,test}}/{{OK,NOK}}/")
        if len(sel) < len(rep):
            print(f"  {len(rep) - len(sel)} images left out; see the report.")


if __name__ == "__main__":
    main()

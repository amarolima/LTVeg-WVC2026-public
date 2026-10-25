"""Shared utilities for the bootstrap retraining experiments (ResNet18 and YOLO26)."""
import csv
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

CLASSES = ["OK", "NOK"]


def list_class_files(split_dir: Path):
    """Returns {class_name: [file_paths]} for a train/val/test split directory."""
    out = {}
    for cls in CLASSES:
        out[cls] = sorted((split_dir / cls).glob("*.png"))
    return out


def bootstrap_resample(class_files: dict, seed: int):
    """Resamples each class independently, with replacement, to its original size.
    Stratified bootstrap: preserves the OK/NOK ratio of the original training set
    exactly (avoids degenerate resamples where a whole class could vanish).
    """
    rng = random.Random(seed)
    resampled = {}
    for cls, files in class_files.items():
        n = len(files)
        resampled[cls] = [files[rng.randrange(n)] for _ in range(n)]
    return resampled


def metrics_table(labels, preds):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )
    macro_f1 = f1.mean()
    out = {}
    for i, cls in enumerate(CLASSES):
        out[f"{cls}_recall"] = round(100 * recall[i], 4)
        out[f"{cls}_precision"] = round(100 * precision[i], 4)
        out[f"{cls}_f1"] = round(100 * f1[i], 4)
    out["macro_f1"] = round(100 * macro_f1, 4)
    return out


def append_csv_row(csv_path: Path, row: dict):
    is_new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def completed_replicates(csv_path: Path):
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {int(row["replicate"]) for row in csv.DictReader(f)}


def percentile_ci(values, alpha=0.05):
    lo = np.percentile(values, 100 * alpha / 2)
    hi = np.percentile(values, 100 * (1 - alpha / 2))
    # ddof=1 (desvio amostral), o mesmo usado no artigo: 4,19 e 5,02 -> 4,2 e 5,0.
    # A versao original usava ddof=0 (4,12 e 4,93); so o summary.json muda.
    return float(np.mean(values)), float(np.std(values, ddof=1)), float(lo), float(hi)


def summarize(csv_path: Path, out_json: Path):
    import json
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k: (float(v) if k != "replicate" else int(v)) for k, v in r.items()})
    n = len(rows)
    metric_keys = [k for k in rows[0].keys() if k != "replicate"]
    summary = {"n_replicates": n}
    for k in metric_keys:
        vals = [r[k] for r in rows]
        mean, std, lo, hi = percentile_ci(vals)
        summary[k] = {"mean": round(mean, 2), "std": round(std, 2),
                       "ci95_lo": round(lo, 2), "ci95_hi": round(hi, 2)}
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    return summary

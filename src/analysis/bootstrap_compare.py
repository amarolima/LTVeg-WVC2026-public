"""
Compares the ResNet18 and YOLO26 bootstrap distributions (30 retraining replicates
each, see bootstrap_resnet.py / bootstrap_yolo.py) with a Welch's t-test per metric.
The two samples are independent (different bootstrap resamples of the training set
per model, not paired), so an unpaired test is used.

Usage:
    python src/analysis/bootstrap_compare.py --resnet-dir results/bootstrap_resnet --yolo-dir results/bootstrap_yolo_adamw --out results/bootstrap_comparison.json
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resnet-dir", required=True, type=Path)
    ap.add_argument("--yolo-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    r = pd.read_csv(args.resnet_dir / "results.csv")
    y = pd.read_csv(args.yolo_dir / "results.csv")

    metrics = ["test_macro_f1", "test_OK_recall", "test_OK_precision", "test_OK_f1",
               "test_NOK_recall", "test_NOK_precision", "test_NOK_f1"]

    out = {"n_resnet": len(r), "n_yolo": len(y), "comparisons": {}}
    for col in metrics:
        rv, yv = r[col].values, y[col].values
        t, p = stats.ttest_ind(rv, yv, equal_var=False)
        out["comparisons"][col] = {
            "resnet_mean": round(float(rv.mean()), 2),
            "yolo_mean": round(float(yv.mean()), 2),
            "diff_resnet_minus_yolo": round(float(rv.mean() - yv.mean()), 2),
            "welch_t": round(float(t), 3),
            "p_value": round(float(p), 5),
            "significant_at_0.05": bool(p < 0.05),
        }

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    for col, res in out["comparisons"].items():
        sig = "significant" if res["significant_at_0.05"] else "not significant"
        print(f"{col}: ResNet={res['resnet_mean']}  YOLO={res['yolo_mean']}  "
              f"diff={res['diff_resnet_minus_yolo']}  p={res['p_value']} ({sig})")


if __name__ == "__main__":
    main()

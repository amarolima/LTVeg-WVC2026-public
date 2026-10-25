"""
YOLO26 hyperparameter ablation on the FIXED original split (data/dataset, not
bootstrap-resampled) -- one run per config, to quickly screen whether the gap to
ResNet18 (test macro-F1 ~86.6% single-run / ~83.1% over 30 bootstrap retrains) can be
closed before committing to a full bootstrap re-run of any candidate config.

Ultralytics' classification defaults are tuned for large (ImageNet-scale) datasets:
randaugment, erasing=0.4, scale jitter +-50%, hsv jitter, auto optimizer/lr. On a
210-image fine-tuning set these may be too aggressive and could be actively hurting
convergence relative to ResNet18's much gentler augmentation (horizontal flip only).
This script tests that hypothesis plus a few standard small-dataset knobs (explicit
low LR, dropout, smaller image size, smaller backbone).

Usage:
    python src/training/yolo_ablation.py --data-dir data/dataset --out-dir results/yolo_ablation
"""
import argparse
import gc
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_common import CLASSES, append_csv_row, list_class_files, metrics_table

PATIENCE = 30  # more generous than the bootstrap's 15, since these are one-off runs

CONFIGS = {
    "baseline_current": dict(model="yolo26s-cls.pt", imgsz=640, batch=16),
    "light_aug": dict(model="yolo26s-cls.pt", imgsz=640, batch=16,
                       auto_augment=None, erasing=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                       degrees=0.0, translate=0.0, scale=0.0, mixup=0.0),
    "explicit_adamw": dict(model="yolo26s-cls.pt", imgsz=640, batch=16,
                            optimizer="AdamW", lr0=1e-4, cos_lr=True),
    "light_aug_adamw": dict(model="yolo26s-cls.pt", imgsz=640, batch=16,
                             auto_augment=None, erasing=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                             degrees=0.0, translate=0.0, scale=0.0, mixup=0.0,
                             optimizer="AdamW", lr0=1e-4, cos_lr=True),
    "light_aug_adamw_dropout": dict(model="yolo26s-cls.pt", imgsz=640, batch=16,
                                     auto_augment=None, erasing=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                                     degrees=0.0, translate=0.0, scale=0.0, mixup=0.0,
                                     optimizer="AdamW", lr0=1e-4, cos_lr=True, dropout=0.2),
    "light_aug_adamw_imgsz224": dict(model="yolo26s-cls.pt", imgsz=224, batch=16,
                                      auto_augment=None, erasing=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                                      degrees=0.0, translate=0.0, scale=0.0, mixup=0.0,
                                      optimizer="AdamW", lr0=1e-4, cos_lr=True),
    "light_aug_adamw_nano": dict(model="yolo26n-cls.pt", imgsz=640, batch=16,
                                  auto_augment=None, erasing=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
                                  degrees=0.0, translate=0.0, scale=0.0, mixup=0.0,
                                  optimizer="AdamW", lr0=1e-4, cos_lr=True),
}


def predict_dir(model, split_dir: Path, class_to_idx: dict, imgsz: int):
    model_name_to_idx = {name: idx for idx, name in model.names.items()}
    our_idx_to_model_idx = {our_idx: model_name_to_idx[cls] for cls, our_idx in class_to_idx.items()}
    model_idx_to_our_idx = {v: k for k, v in our_idx_to_model_idx.items()}

    files = list_class_files(split_dir)
    paths = [p for c, ps in files.items() for p in ps]
    labels = [class_to_idx[c] for c, ps in files.items() for p in ps]
    results = model.predict(source=[str(p) for p in paths], imgsz=imgsz, device=0, verbose=False)
    preds = [model_idx_to_our_idx[int(r.probs.top1)] for r in results]
    import numpy as np
    return np.array(labels), np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--only", default=None, help="comma-separated subset of config names to run")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "results.csv"
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    names = list(CONFIGS.keys())
    if args.only:
        names = [n for n in names if n in set(args.only.split(","))]

    for name in names:
        cfg = dict(CONFIGS[name])
        model_weights = cfg.pop("model")
        imgsz = cfg["imgsz"]
        print(f"\n=== YOLO26 ablation: {name} ({model_weights}, {cfg}) ===")

        model = YOLO(model_weights)
        run_dir = (args.out_dir / "runs").resolve()
        model.train(
            data=str(args.data_dir.resolve()), epochs=100, patience=PATIENCE,
            batch=cfg.pop("batch"), imgsz=cfg.pop("imgsz"), workers=2, device=0,
            project=str(run_dir), name=name, exist_ok=True,
            verbose=False, plots=False, seed=42,
            **cfg,
        )

        best_weights = run_dir / name / "weights" / "best.pt"
        eval_model = YOLO(str(best_weights))

        val_labels, val_preds = predict_dir(eval_model, args.data_dir / "val", class_to_idx, imgsz)
        val_metrics = metrics_table(val_labels, val_preds)
        test_labels, test_preds = predict_dir(eval_model, args.data_dir / "test", class_to_idx, imgsz)
        test_metrics = metrics_table(test_labels, test_preds)

        row = {"config": name, "val_macro_f1": val_metrics["macro_f1"]}
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        append_csv_row(csv_path, row)
        print(f"  val_macro_f1={val_metrics['macro_f1']:.2f}  test_macro_f1={test_metrics['macro_f1']:.2f}  "
              f"test_NOK_recall={test_metrics['NOK_recall']:.2f}")

        del model, eval_model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nAll done. Results in {csv_path}")


if __name__ == "__main__":
    main()

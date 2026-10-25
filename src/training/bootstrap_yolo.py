"""
Bootstrap retraining experiment for the YOLO26 classification model.

Same protocol as bootstrap_resnet.py: for each of N replicates, resample the
training set with replacement (stratified per class, same size as the original
210 images) and train YOLO26s-cls from scratch on it; validation and test sets
are fixed (never resampled). Because Ultralytics requires an on-disk
ImageFolder-style directory, each replicate's resampled training set is
materialized as a temp directory of symlinks (val/test are symlinked once and
reused for all replicates).

Resumable: skips replicates already present in the output CSV.

Usage:
    python src/training/bootstrap_yolo.py --data-dir data/dataset --out-dir results/bootstrap_yolo_adamw --n 30 --config explicit_adamw
"""
import argparse
import gc
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_common import (CLASSES, append_csv_row, bootstrap_resample,
                               completed_replicates, list_class_files,
                               metrics_table, summarize)
from yolo_ablation import CONFIGS

MAX_EPOCHS = 100
PATIENCE = 15


def materialize_split(class_files: dict, split_dir: Path):
    split_dir.mkdir(parents=True, exist_ok=True)
    for cls, files in class_files.items():
        cls_dir = split_dir / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for j, src in enumerate(files):
            dst = cls_dir / f"{j}_{src.name}"
            if not dst.exists():
                dst.symlink_to(src.resolve())


@torch.no_grad()
def predict_dir(model, split_dir: Path, class_to_idx: dict, device, imgsz):
    # Ultralytics assigns its own internal class indices alphabetically by folder
    # name (NOK=0, OK=1), independent of our fixed CLASSES order (OK=0, NOK=1).
    # Predictions must be remapped through model.names before comparing to our labels.
    model_name_to_idx = {name: idx for idx, name in model.names.items()}
    our_idx_to_model_idx = {our_idx: model_name_to_idx[cls] for cls, our_idx in class_to_idx.items()}
    model_idx_to_our_idx = {v: k for k, v in our_idx_to_model_idx.items()}

    files = list_class_files(split_dir)
    paths = [p for c, ps in files.items() for p in ps]
    labels = [class_to_idx[c] for c, ps in files.items() for p in ps]
    results = model.predict(source=[str(p) for p in paths], imgsz=imgsz,
                             device=device, verbose=False)
    preds = [model_idx_to_our_idx[int(r.probs.top1)] for r in results]
    return np.array(labels), np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed-offset", type=int, default=0)
    ap.add_argument("--config", default="baseline_current", choices=list(CONFIGS.keys()),
                     help="hyperparameter preset from yolo_ablation.CONFIGS")
    args = ap.parse_args()

    cfg = dict(CONFIGS[args.config])
    model_weights = cfg.pop("model")
    img_size = cfg.pop("imgsz")
    batch_size = cfg.pop("batch")
    print(f"Using config '{args.config}': model={model_weights} imgsz={img_size} batch={batch_size} extra={cfg}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "results.csv"
    done = completed_replicates(csv_path)
    print(f"{len(done)} replicates already completed, {args.n - len(done)} remaining")

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    train_files = list_class_files(args.data_dir / "train")
    val_files = list_class_files(args.data_dir / "val")
    test_files = list_class_files(args.data_dir / "test")

    work_dir = args.out_dir.resolve() / "work"
    fixed_dir = work_dir / "fixed"
    materialize_split(val_files, fixed_dir / "val")
    materialize_split(test_files, fixed_dir / "test")

    for i in range(args.n):
        if i in done:
            continue
        seed = 2000 + args.seed_offset + i
        print(f"\n=== YOLO26 bootstrap replicate {i} (seed={seed}) ===")

        rep_dir = work_dir / f"rep{i}"
        resampled = bootstrap_resample(train_files, seed=seed)
        materialize_split(resampled, rep_dir / "train")
        # symlink the fixed val set into this replicate's data dir (ultralytics needs train+val together)
        (rep_dir / "val").symlink_to((fixed_dir / "val").resolve())

        model = YOLO(model_weights)
        run_name = f"rep{i}"
        model.train(
            data=str(rep_dir), epochs=MAX_EPOCHS, patience=PATIENCE,
            imgsz=img_size, batch=batch_size, workers=2, device=0,
            project=str(work_dir / "runs"), name=run_name, exist_ok=True,
            verbose=False, plots=False, seed=seed,
            **cfg,
        )

        best_weights = work_dir / "runs" / run_name / "weights" / "best.pt"
        eval_model = YOLO(str(best_weights))

        val_labels, val_preds = predict_dir(eval_model, fixed_dir / "val", class_to_idx, device=0, imgsz=img_size)
        val_metrics = metrics_table(val_labels, val_preds)

        test_labels, test_preds = predict_dir(eval_model, fixed_dir / "test", class_to_idx, device=0, imgsz=img_size)
        test_metrics = metrics_table(test_labels, test_preds)

        row = {"replicate": i, "val_macro_f1": val_metrics["macro_f1"]}
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        append_csv_row(csv_path, row)
        print(f"  val_macro_f1={val_metrics['macro_f1']:.2f}  test_macro_f1={test_metrics['macro_f1']:.2f}")

        del model, eval_model
        gc.collect()
        torch.cuda.empty_cache()
        # free disk: drop this replicate's symlink tree and heavy run artifacts, keep only metrics
        shutil.rmtree(rep_dir, ignore_errors=True)
        shutil.rmtree(work_dir / "runs" / run_name, ignore_errors=True)

    summary = summarize(csv_path, args.out_dir / "summary.json")
    print("\n=== SUMMARY (YOLO26, N replicates) ===")
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()

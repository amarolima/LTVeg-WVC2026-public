"""
Bootstrap retraining experiment for the ResNet18 baseline.

For each of N replicates: resample the training set with replacement (stratified
per class, same size as the original 210 images), train a fresh ResNet18 from
ImageNet weights, and evaluate on the FIXED validation/test sets (the same ones
used throughout the paper). Validation and test sets are never resampled, only
the training set is -- this measures how sensitive the trained model's test
performance is to which particular training images it happened to see, which is
the relevant question given only 351 labeled images are available.

Resumable: skips replicates already present in the output CSV.

Usage:
    python src/training/bootstrap_resnet.py --data-dir data/dataset --out-dir results/bootstrap_resnet --n 30
"""
import argparse
import gc
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_common import (CLASSES, append_csv_row, bootstrap_resample,
                               completed_replicates, list_class_files,
                               metrics_table, summarize)

IMG_SIZE = 640
BATCH_SIZE = 16
MAX_EPOCHS = 100
PATIENCE = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4


class FileListDataset(Dataset):
    def __init__(self, samples, transform):
        # samples: list of (path, label_idx)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, eval_tf


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        preds = model(x).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def train_one_replicate(train_samples, val_loader, device, seed):
    torch.manual_seed(seed)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model = model.to(device)

    train_tf, _ = build_transforms()
    train_ds = FileListDataset(train_samples, train_tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_labels, val_preds = evaluate(model, val_loader, device)
        val_metrics = metrics_table(val_labels, val_preds)
        val_f1 = val_metrics["macro_f1"]
        scheduler.step(100 - val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                break

    model.load_state_dict(best_state)
    del train_loader, train_ds
    return model, best_val_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "results.csv"
    done = completed_replicates(csv_path)
    print(f"{len(done)} replicates already completed, {args.n - len(done)} remaining")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    train_files = list_class_files(args.data_dir / "train")
    _, eval_tf = build_transforms()

    def make_fixed_loader(split):
        files = list_class_files(args.data_dir / split)
        samples = [(p, class_to_idx[c]) for c, ps in files.items() for p in ps]
        ds = FileListDataset(samples, eval_tf)
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    val_loader = make_fixed_loader("val")
    test_loader = make_fixed_loader("test")

    for i in range(args.n):
        if i in done:
            continue
        seed = 1000 + args.seed_offset + i
        print(f"\n=== ResNet18 bootstrap replicate {i} (seed={seed}) ===")

        resampled = bootstrap_resample(train_files, seed=seed)
        train_samples = [(p, class_to_idx[c]) for c, ps in resampled.items() for p in ps]

        model, best_val_f1 = train_one_replicate(train_samples, val_loader, device, seed)

        test_labels, test_preds = evaluate(model, test_loader, device)
        test_metrics = metrics_table(test_labels, test_preds)

        row = {"replicate": i, "val_macro_f1": round(best_val_f1, 4)}
        row.update({f"test_{k}": v for k, v in test_metrics.items()})
        append_csv_row(csv_path, row)
        print(f"  val_macro_f1={best_val_f1:.2f}  test_macro_f1={test_metrics['macro_f1']:.2f}")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    summary = summarize(csv_path, args.out_dir / "summary.json")
    print("\n=== SUMMARY (ResNet18, N replicates) ===")
    for k, v in summary.items():
        print(k, v)


if __name__ == "__main__":
    main()

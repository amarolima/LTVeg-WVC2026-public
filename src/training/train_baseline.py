"""
Non-YOLO baseline for the ROW vegetation classification paper (CTML 2026 revision).

Trains a ResNet18 (ImageNet-pretrained, fine-tuned) on the same OK/NOK train/val/test
split already used for the YOLO26 model (data/dataset/{train,val,test}/{OK,NOK}),
and reports the same metrics used in the paper's tables: recall, precision, F1 and
error per class, plus macro-F1, on both validation and test sets. Also saves
confusion-matrix figures in the same visual style as Figuras/Resultados/matriz_confusao_*.

Usage:
    python src/training/train_baseline.py --data-dir data/dataset --out-dir results/train_baseline_rerun
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt

SEED = 42
IMG_SIZE = 640
BATCH_SIZE = 16
MAX_EPOCHS = 100
PATIENCE = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4

CLASSES = ["OK", "NOK"]  # fixed order, matches the paper's tables


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(data_dir: Path):
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

    class_to_idx = {c: i for i, c in enumerate(CLASSES)}

    def make_ds(split, tf):
        ds = datasets.ImageFolder(str(data_dir / split), transform=tf)
        # remap ImageFolder's alphabetical class_to_idx (NOK=0,OK=1) to our fixed order
        remap = {ds.class_to_idx[c]: class_to_idx[c] for c in CLASSES}
        ds.samples = [(p, remap[l]) for p, l in ds.samples]
        ds.targets = [remap[l] for l in ds.targets]
        ds.class_to_idx = class_to_idx
        return ds

    train_ds = make_ds("train", train_tf)
    val_ds = make_ds("val", eval_tf)
    test_ds = make_ds("test", eval_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, len(train_ds), len(val_ds), len(test_ds)


def build_model(device):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    return np.array(all_labels), np.array(all_preds), avg_loss


def metrics_table(labels, preds):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=[0, 1], zero_division=0
    )
    macro_f1 = f1.mean()
    out = {}
    for i, cls in enumerate(CLASSES):
        out[cls] = {
            "recall": round(100 * recall[i], 2),
            "precision": round(100 * precision[i], 2),
            "f1": round(100 * f1[i], 2),
            "error": round(100 * (1 - recall[i]), 2),
            "support": int(support[i]),
        }
    out["macro_f1"] = round(100 * macro_f1, 2)
    return out


def save_confusion_matrix(labels, preds, title, out_path):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(CLASSES)
    ax.set_yticks([0, 1]); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, n_train, n_val, n_test = build_dataloaders(args.data_dir)
    print(f"train={n_train} val={n_val} test={n_test}")

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_accuracy_top1": []}

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / n_train

        val_labels, val_preds, val_loss = evaluate(model, val_loader, device)
        val_metrics = metrics_table(val_labels, val_preds)
        val_f1 = val_metrics["macro_f1"]
        scheduler.step(val_loss)

        val_acc = float((val_preds == val_labels).mean() * 100)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_macro_f1"].append(val_f1)
        history["val_accuracy_top1"].append(val_acc)

        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_macro_f1 {val_f1:.2f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (no val improvement for {PATIENCE} epochs)")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, args.out_dir / "resnet18_best.pt")
    with open(args.out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    val_labels, val_preds, _ = evaluate(model, val_loader, device)
    test_labels, test_preds, _ = evaluate(model, test_loader, device)

    val_metrics = metrics_table(val_labels, val_preds)
    test_metrics = metrics_table(test_labels, test_preds)

    save_confusion_matrix(val_labels, val_preds, "ResNet18 - Validation", args.out_dir / "confusion_matrix_val.png")
    save_confusion_matrix(test_labels, test_preds, "ResNet18 - Test", args.out_dir / "confusion_matrix_test.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].plot(history["train_loss"], label="train_loss")
    axes[0].plot(history["val_loss"], label="val_loss")
    axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(history["val_macro_f1"], label="val_macro_f1", color="green")
    axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].set_title("Validation macro-F1 (%)")
    fig.tight_layout()
    fig.savefig(args.out_dir / "training_curves.png", dpi=200)
    plt.close(fig)

    results = {
        "model": "ResNet18 (ImageNet-pretrained, fine-tuned)",
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "best_epoch_val_macro_f1": best_val_f1,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "validation": val_metrics,
        "test": test_metrics,
    }
    with open(args.out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== VALIDATION ===")
    print(json.dumps(val_metrics, indent=2))
    print("\n=== TEST ===")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()

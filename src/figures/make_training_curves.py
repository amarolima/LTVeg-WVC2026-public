"""
Regenerates the 6 small training-curve panels (train_loss, accuracy_top1, val_loss
for each of ResNet18 and YOLO26+AdamW) used in the paper, from the representative
single training runs already on disk:
  - ResNet18: results/train_baseline/history.json (train_baseline.py run)
  - YOLO26+AdamW: results/yolo_ablation/explicit_adamw_epochs.csv
These figures were dropped from the final WVC paper (single-run curves).

Usage (from the repository root):
    python src/figures/make_training_curves.py
"""
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
OUT_DIR = ROOT / "results" / "figuras_extras"


def single_panel(x, y, title, ylabel, color, out_path):
    fig, ax = plt.subplots(figsize=(2.6, 2.2))
    ax.plot(x, y, color=color)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


def resnet_curves():
    history = json.loads((ROOT / "results" / "train_baseline" / "history.json").read_text())
    epochs = list(range(1, len(history["train_loss"]) + 1))
    single_panel(epochs, history["train_loss"], "Train loss", "loss", "tab:blue",
                 OUT_DIR / "resnet_train_loss.png")
    single_panel(epochs, history["val_accuracy_top1"], "Val. accuracy (top-1)", "%", "tab:orange",
                 OUT_DIR / "resnet_accuracy_top1.png")
    single_panel(epochs, history["val_loss"], "Validation loss", "loss", "tab:green",
                 OUT_DIR / "resnet_val_loss.png")


def yolo_curves():
    df = pd.read_csv(ROOT / "results" / "yolo_ablation" / "explicit_adamw_epochs.csv")
    epochs = df["epoch"].tolist()
    single_panel(epochs, df["train/loss"], "Train loss", "loss", "tab:blue",
                 OUT_DIR / "yolo_train_loss.png")
    single_panel(epochs, df["metrics/accuracy_top1"] * 100, "Val. accuracy (top-1)", "%", "tab:orange",
                 OUT_DIR / "yolo_accuracy_top1.png")
    single_panel(epochs, df["val/loss"], "Validation loss", "loss", "tab:green",
                 OUT_DIR / "yolo_val_loss.png")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    resnet_curves()
    yolo_curves()

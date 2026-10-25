"""
Regenerates confusion-matrix figures for the paper, matching the original visual
style (order NOK,OK; title 'Confusion Matrix'), for the two models now being
reported: ResNet18 (single representative run,
checkpoints/resnet18_best.pt) and YOLO26+AdamW (single representative run, from the
ablation weights, checkpoints/yolo26s_adamw_best.pt).
These figures were dropped from the final WVC paper (6-page limit).

Usage (from the repository root):
    python src/figures/make_paper_confusion_matrices.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import confusion_matrix
from torchvision import models, transforms
import matplotlib.pyplot as plt
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from bootstrap_common import list_class_files

PLOT_ORDER = ["NOK", "OK"]  # matches the paper's original figure style


def save_matrix(labels_names, preds_names, title, out_path):
    # sklearn convention: rows = true, columns = predicted. Below, cm[i, j] is
    # drawn at ax.text(j, i), so the row index i lands on the y-axis -- hence
    # y-axis = True and x-axis = Predicted. (v1 and v2 of the paper had these
    # two labels swapped.)
    cm = confusion_matrix(labels_names, preds_names, labels=PLOT_ORDER)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(PLOT_ORDER)
    ax.set_yticks([0, 1]); ax.set_yticklabels(PLOT_ORDER)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")


def resnet_predictions(weights_path, data_dir, split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()

    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor(),
                              transforms.Normalize(mean, std)])
    class_order = ["OK", "NOK"]  # matches train_baseline.py's CLASSES order used when saving state
    files = list_class_files(data_dir / split)
    labels_names, preds_names = [], []
    with torch.no_grad():
        for cls in class_order:
            for p in files[cls]:
                img = Image.open(p).convert("RGB")
                x = tf(img).unsqueeze(0).to(device)
                pred_idx = model(x).argmax(dim=1).item()
                labels_names.append(cls)
                preds_names.append(class_order[pred_idx])
    return labels_names, preds_names


def yolo_predictions(weights_path, data_dir, split, imgsz=640):
    model = YOLO(str(weights_path))
    model_idx_to_name = model.names
    files = list_class_files(data_dir / split)
    labels_names, preds_names = [], []
    for cls, paths in files.items():
        results = model.predict(source=[str(p) for p in paths], imgsz=imgsz, device=0, verbose=False)
        for r in results:
            labels_names.append(cls)
            preds_names.append(model_idx_to_name[int(r.probs.top1)])
    return labels_names, preds_names


def main():
    root = Path(".")
    data_dir = root / "data/dataset"
    out_dir = root / "results" / "figuras_extras"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ResNet18
    for split, tag in [("val", "validacao"), ("test", "teste")]:
        labels_names, preds_names = resnet_predictions(
            root / "checkpoints" / "resnet18_best.pt", data_dir, split)
        save_matrix(labels_names, preds_names, "Confusion Matrix",
                    out_dir / f"matriz_confusao_{tag}_resnet.png")

    # YOLO26 + AdamW
    for split, tag in [("val", "validacao"), ("test", "teste")]:
        labels_names, preds_names = yolo_predictions(
            root / "checkpoints" / "yolo26s_adamw_best.pt",
            data_dir, split)
        save_matrix(labels_names, preds_names, "Confusion Matrix",
                    out_dir / f"matriz_confusao_{tag}_yolo_adamw.png")


if __name__ == "__main__":
    main()

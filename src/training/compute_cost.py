"""
Computational cost comparison between ResNet18 and YOLO26 (AdamW config, the one
reported in the paper), on the same hardware (RTX 4050 Laptop GPU) used for all
other experiments in this project.

Measures, for both models:
  - parameter count
  - checkpoint size on disk (MB)
  - GFLOPs at the 640x640 input resolution used throughout this work
  - inference latency (ms/image, batch=1, GPU, averaged over repeated runs after warmup)
  - wall-clock time per training epoch (train+val pass on the actual 210/72 split),
    averaged over 3 epochs, same protocol as the rest of the project (AdamW lr=1e-4,
    batch=16, imgsz=640)

Usage:
    python src/training/compute_cost.py --data-dir data/dataset --out results/compute_cost.json
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from thop import profile
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from ultralytics import YOLO

IMG_SIZE = 640
BATCH_SIZE = 16
N_EPOCHS_TIMED = 3
N_LATENCY_RUNS = 50
N_LATENCY_WARMUP = 10


class FileListDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def build_resnet_loaders(data_dir: Path):
    from torchvision.datasets import ImageFolder
    mean = [0.485, 0.456, 0.406]; std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
                              transforms.Normalize(mean, std)])
    train_ds = ImageFolder(str(data_dir / "train"), transform=tf)
    val_ds = ImageFolder(str(data_dir / "val"), transform=tf)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader


def resnet_epoch_time(data_dir: Path, device):
    train_loader, val_loader = build_resnet_loaders(data_dir)
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    times = []
    for epoch in range(N_EPOCHS_TIMED):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                model(x)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return sum(times) / len(times), model


def resnet_static_cost(model, device):
    n_params = sum(p.numel() for p in model.parameters())
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    macs, _ = profile(model, inputs=(dummy,), verbose=False)
    gflops = 2 * macs / 1e9  # MACs -> FLOPs
    return n_params, gflops


def resnet_latency(model, device):
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        for _ in range(N_LATENCY_WARMUP):
            model(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_LATENCY_RUNS):
            model(dummy)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
    return 1000 * total / N_LATENCY_RUNS  # ms/image


def yolo_epoch_time(data_dir: Path):
    model = YOLO("yolo26s-cls.pt")
    t0 = time.perf_counter()
    model.train(
        data=str(data_dir.resolve()), epochs=N_EPOCHS_TIMED, patience=N_EPOCHS_TIMED + 1,
        imgsz=IMG_SIZE, batch=BATCH_SIZE, workers=2, device=0,
        optimizer="AdamW", lr0=1e-4, cos_lr=True,
        project="/tmp/yolo_cost_timing", name="t", exist_ok=True,
        verbose=False, plots=False, seed=42,
    )
    total = time.perf_counter() - t0
    return total / N_EPOCHS_TIMED, model


def yolo_static_cost(model, device):
    net = model.model.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    net.eval()
    macs, _ = profile(net, inputs=(dummy,), verbose=False)
    gflops = 2 * macs / 1e9
    return n_params, gflops


def yolo_latency(model, device):
    net = model.model.to(device)
    net.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    with torch.no_grad():
        for _ in range(N_LATENCY_WARMUP):
            net(dummy)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(N_LATENCY_RUNS):
            net(dummy)
        torch.cuda.synchronize()
        total = time.perf_counter() - t0
    return 1000 * total / N_LATENCY_RUNS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    print("=== ResNet18 ===")
    epoch_time_r, resnet_model = resnet_epoch_time(args.data_dir, device)
    n_params_r, gflops_r = resnet_static_cost(resnet_model, device)
    latency_r = resnet_latency(resnet_model, device)
    resnet_ckpt = Path("checkpoints/resnet18_best.pt")
    size_r = resnet_ckpt.stat().st_size / 1e6 if resnet_ckpt.exists() else None
    results["resnet18"] = {
        "params": n_params_r, "params_M": round(n_params_r / 1e6, 2),
        "gflops": round(gflops_r, 2), "checkpoint_size_MB": round(size_r, 2) if size_r else None,
        "inference_latency_ms": round(latency_r, 3),
        "epoch_time_s": round(epoch_time_r, 2),
    }
    print(results["resnet18"])
    del resnet_model
    torch.cuda.empty_cache()

    print("=== YOLO26 (AdamW) ===")
    epoch_time_y, yolo_model = yolo_epoch_time(args.data_dir)
    n_params_y, gflops_y = yolo_static_cost(yolo_model, device)
    latency_y = yolo_latency(yolo_model, device)
    yolo_ckpt = Path("/tmp/yolo_cost_timing/t/weights/best.pt")
    size_y = yolo_ckpt.stat().st_size / 1e6 if yolo_ckpt.exists() else None
    results["yolo26"] = {
        "params": int(n_params_y), "params_M": round(n_params_y / 1e6, 2),
        "gflops": round(gflops_y, 2), "checkpoint_size_MB": round(size_y, 2) if size_y else None,
        "inference_latency_ms": round(latency_y, 3),
        "epoch_time_s": round(epoch_time_y, 2),
    }
    print(results["yolo26"])

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()

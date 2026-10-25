"""
Analise de limiar da classe NOK + figura de erros qualitativos.

Roda inferencia UMA vez com cada modelo no conjunto de teste, guarda as
probabilidades, e a partir delas gera:
  1. tabela de varredura de limiar (CSV + markdown)
  2. curva precision-recall da classe NOK (PNG) com AP e ROC-AUC
  3. figura com os erros de classificacao (falsos negativos e falsos positivos)

Nao retreina nada -- usa os pesos ja salvos.

Uso (a partir da RAIZ do projeto, mesmo lugar de onde voce roda os outros):
    conda run -n research python src/analysis/nok_analysis.py
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score)
from torchvision import models, transforms
from ultralytics import YOLO

ROOT = Path(".")
DATA = ROOT / "data/dataset"
OUT = ROOT / "results" / "nok_analysis"
CLASSES = ["OK", "NOK"]          # convencao do projeto: OK=0, NOK=1
IMG_SIZE = 640
SPLIT = "test"

RESNET_W = ROOT / "checkpoints" / "resnet18_best.pt"
YOLO_W = ROOT / "checkpoints" / "yolo26s_adamw_best.pt"


def list_split(split):
    """Retorna (paths, y) com y=1 para NOK e y=0 para OK."""
    paths, y = [], []
    for cls in CLASSES:
        for p in sorted((DATA / split / cls).glob("*.png")):
            paths.append(p)
            y.append(CLASSES.index(cls))
    return paths, np.array(y)


def resnet_scores(weights, paths):
    """P(NOK) do ResNet18. Indice 1 = NOK (ordem de train_baseline.py)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(weights, map_location=dev))
    m = m.to(dev).eval()
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    out = []
    with torch.no_grad():
        for p in paths:
            x = tf(Image.open(p).convert("RGB")).unsqueeze(0).to(dev)
            out.append(float(torch.softmax(m(x), dim=1)[0, CLASSES.index("NOK")]))
    return np.array(out)


def yolo_scores(weights, paths):
    """P(NOK) do YOLO26. ATENCAO: o Ultralytics ordena as classes
    alfabeticamente (NOK=0, OK=1), invertido em relacao a nossa convencao --
    o mesmo bug que ja foi corrigido em bootstrap_yolo.py. Remapeamos via
    model.names em vez de assumir indice."""
    m = YOLO(str(weights))
    name_to_idx = {name: idx for idx, name in m.names.items()}
    nok_idx = name_to_idx["NOK"]
    dev = 0 if torch.cuda.is_available() else "cpu"
    res = m.predict(source=[str(p) for p in paths], imgsz=IMG_SIZE,
                    device=dev, verbose=False)
    return np.array([float(r.probs.data[nok_idx]) for r in res])


def sweep_table(y, s):
    """Para cada limiar candidato, quantos NOK achamos e quantas inspecoes custa."""
    n_pos = int(y.sum())
    rows = []
    for t in sorted(set(np.round(s, 4)), reverse=True):
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = n_pos - tp
        rec = tp / n_pos
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rows.append(dict(threshold=t, nok_recall=100 * rec, nok_precision=100 * prec,
                         inspecoes=tp + fp, nok_perdidos=fn))
    return rows


def analyse(name, y, s, ax):
    prec, rec, _ = precision_recall_curve(y, s)          # positivo = NOK
    ap = average_precision_score(y, s)
    auc = roc_auc_score(y, s)

    pred = (s >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(y.sum()) - tp
    rec05 = tp / y.sum()
    prec05 = tp / (tp + fp) if (tp + fp) else 0.0
    r05, p05 = 100 * rec05, 100 * prec05

    ax.plot(rec, prec, lw=2, label=f"{name} (AP={ap:.3f})")
    ax.scatter([rec05], [prec05], marker="o", s=60, zorder=5)

    print(f"\n{'='*66}\n{name}")
    print(f"  AP (average precision, classe NOK) = {ap:.4f}")
    print(f"  ROC-AUC                            = {auc:.4f}")
    print(f"  ponto de operacao padrao (limiar 0.5):")
    print(f"    NOK recall={r05:.1f}%  NOK precision={p05:.1f}%  perdidos={fn}/{int(y.sum())}")

    rows = sweep_table(y, s)
    print(f"\n  {'limiar':>8} {'recall':>8} {'precis':>8} {'inspec':>7} {'perdidos':>9}")
    seen = set()
    for r in rows:
        key = (round(r["nok_recall"], 1), r["inspecoes"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  {r['threshold']:8.4f} {r['nok_recall']:7.1f}% {r['nok_precision']:7.1f}%"
              f" {r['inspecoes']:7d} {r['nok_perdidos']:9d}")

    import csv
    with open(OUT / f"sweep_{name.lower().replace(' ', '_')}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return dict(name=name, ap=float(ap), roc_auc=float(auc),
                recall_at_05=float(r05), precision_at_05=float(p05))


def error_figure(name, paths, y, s, out_path, max_per_kind=3):
    """Mostra falsos negativos (NOK que passou como OK) e falsos positivos."""
    pred = (s >= 0.5).astype(int)
    fn_idx = [i for i in range(len(y)) if y[i] == 1 and pred[i] == 0]
    fp_idx = [i for i in range(len(y)) if y[i] == 0 and pred[i] == 1]
    # mais "confiantes" primeiro: os erros que o modelo cometeu com mais certeza
    fn_idx = sorted(fn_idx, key=lambda i: s[i])[:max_per_kind]
    fp_idx = sorted(fp_idx, key=lambda i: -s[i])[:max_per_kind]

    sel = [(i, "FN") for i in fn_idx] + [(i, "FP") for i in fp_idx]
    if not sel:
        print(f"  ({name}: sem erros para plotar)")
        return
    n = len(sel)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.1))
    axes = np.atleast_1d(axes)
    for ax, (i, kind) in zip(axes, sel):
        ax.imshow(Image.open(paths[i]).convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        true_c = CLASSES[y[i]]
        pred_c = CLASSES[pred[i]]
        ax.set_title(f"{kind}: true {true_c} / pred {pred_c}\nP(NOK)={s[i]:.2f}", fontsize=8)
    fig.suptitle(f"{name} — test-set errors", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  salvo {out_path}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    paths, y = list_split(SPLIT)
    print(f"{SPLIT}: {len(paths)} imagens  ({int((y==0).sum())} OK, {int((y==1).sum())} NOK)")
    if len(paths) != 69:
        print(f"  *** ATENCAO: esperava 69 imagens, achei {len(paths)}."
              f" Conferir data/dataset antes de usar estes numeros. ***")

    print("\ninferencia ResNet18...")
    s_res = resnet_scores(RESNET_W, paths)
    print("inferencia YOLO26...")
    s_yolo = yolo_scores(YOLO_W, paths)
    np.savez(OUT / "scores_test.npz", paths=[str(p) for p in paths], y=y,
             resnet=s_res, yolo=s_yolo)

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    summ = [analyse("ResNet18", y, s_res, ax), analyse("YOLO26", y, s_yolo, ax)]
    base = y.mean()
    ax.axhline(base, ls=":", c="gray", lw=1)
    ax.text(0.02, base + 0.02, f"prevalência NOK = {base:.2f}", fontsize=7, color="gray")
    ax.set_xlabel("NOK recall"); ax.set_ylabel("NOK precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "pr_curve_nok.png", dpi=200)
    plt.close(fig)
    print(f"\nsalvo {OUT / 'pr_curve_nok.png'}")

    print("\nfiguras de erro:")
    error_figure("ResNet18", paths, y, s_res, OUT / "errors_resnet.png")
    error_figure("YOLO26", paths, y, s_yolo, OUT / "errors_yolo.png")

    import json
    with open(OUT / "summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    print(f"\nOK. Tudo em {OUT}/")


if __name__ == "__main__":
    main()

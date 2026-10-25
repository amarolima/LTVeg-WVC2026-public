"""
Regera a curva PR e a figura de erros dimensionadas para o tamanho FINAL que
ocupam no artigo do WVC, em vez de gerar grande e deixar o LaTeX reduzir.

Motivo: a versao original da curva PR tinha 4,2 pol de largura e era inserida a
0,70\\columnwidth (~2,45 pol), reduzindo a fonte de 8 pt para ~4,7 pt efetivos --
ilegivel no papel. Aqui as figuras ja nascem no tamanho de uso, com fonte que
resulta em ~7-8 pt efetivos.

Uso (a partir da RAIZ do projeto):
    conda run -n research python src/figures/refazer_figuras_wvc.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(".")
SCORES = ROOT / "results" / "nok_analysis" / "scores_test.npz"
OUT = ROOT / "paper" / "Figuras" / "Resultados"

# larguras finais no artigo (coluna IEEE ~3,5 pol)
LARG_PR = 2.30      # 0.657 * columnwidth
LARG_ERR = 2.00     # 0.571 * columnwidth

ERROS = [("160645_G3_500.png", "FN"), ("167547_G2_600.png", "FN"),
         ("179391_G4_230.png", "FP"), ("172708_G1_800.png", "FP")]


def curva_pr(d):
    y, res, yolo = d["y"], d["resnet"], d["yolo"]
    fig, ax = plt.subplots(figsize=(LARG_PR, LARG_PR * 0.82))
    for nome, s in (("ResNet18", res), ("YOLO26", yolo)):
        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.plot(rec, prec, lw=1.6, label=f"{nome} (AP={ap:.3f})")
        pred = (s >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        ax.scatter([tp / y.sum()], [tp / (tp + fp)], marker="o", s=28, zorder=5)
    base = y.mean()
    ax.axhline(base, ls=":", c="gray", lw=0.9)
    ax.text(0.03, base + 0.03, f"NOK prevalence = {base:.2f}", fontsize=6.5, color="gray")
    ax.set_xlabel("NOK recall", fontsize=9)
    ax.set_ylabel("NOK precision", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "pr_curve_nok.png", dpi=400)
    plt.close(fig)
    print("  pr_curve_nok.png")


def figura_erros(d):
    idx = {Path(str(p)).name: i for i, p in enumerate(d["paths"])}
    y, res, yolo = d["y"], d["resnet"], d["yolo"]
    fig, axes = plt.subplots(2, 2, figsize=(LARG_ERR, LARG_ERR * 1.22))
    for ax, (fn, kind) in zip(axes.ravel(), ERROS):
        i = idx[fn]
        ax.imshow(Image.open(str(d["paths"][i])).convert("RGB"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{kind} (true {['OK','NOK'][y[i]]})\n"
                     f"R {res[i]:.2f} | Y {yolo[i]:.2f}", fontsize=7, pad=2)
    fig.tight_layout(pad=0.25, h_pad=1.0)
    fig.savefig(OUT / "erros_qualitativos.png", dpi=400)
    plt.close(fig)
    print("  erros_qualitativos.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(SCORES, allow_pickle=True)
    print(f"regerando em {OUT}/")
    curva_pr(d)
    figura_erros(d)

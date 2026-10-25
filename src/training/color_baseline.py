"""
Baseline classico de indice de cor para a classe NOK.

Pergunta que este script responde: *quanto* do desempenho das CNNs vem de
aprender a tarefa, e quanto vem simplesmente de medir o verde global da cena?

A analise de erros (src/analysis/nok_analysis.py) mostrou que os falsos negativos sao
cenas secas e os falsos positivos sao cenas muito verdes, com as duas
arquiteturas concordando em 89,9% das predicoes. Se esse padrao for a regra, um
limiar sobre um indice de cor trivial deve recuperar boa parte do desempenho --
e nesse caso o artigo nao pode afirmar que as redes agregam valor sem medir isto.

Protocolo IGUAL ao dos modelos profundos, para os numeros serem comparaveis:
  - 30 replicas, reamostragem bootstrap estratificada do TREINO (seeds 3000+i,
    distintas das do ResNet18 (1000+i) e do YOLO26 (2000+i))
  - o limiar (ou a regressao logistica) e ajustado na reamostra de treino
  - validacao e teste permanecem fixos
  - metrica primaria: macro-F1 no teste, media +- desvio (ddof=1)

Alem disso, com um ajuste unico no treino completo, reporta AP e ROC-AUC no
teste -- diretamente comparaveis aos 0,842/0,863 e 0,935/0,943 das CNNs.

Uso (a partir da RAIZ do projeto):
    conda run -n research python src/training/color_baseline.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_common import CLASSES, list_class_files, bootstrap_resample, metrics_table

ROOT = Path(".")
DATA = ROOT / "data/dataset"
OUT = ROOT / "results" / "color_baseline"
N_REP = 30
SEED_BASE = 3000

# nomes das features, na ordem em que extract() as devolve
FEATURES = ["exg_mean", "exg_std", "exg_frac", "lab_a_mean", "lab_a_frac", "green_ratio"]

# variantes avaliadas. "sign" = +1 se valores ALTOS indicam NOK, -1 caso contrario.
SINGLE = [("exg_mean", +1), ("exg_frac", +1), ("lab_a_mean", -1), ("green_ratio", +1)]


def extract(path):
    """Seis estatisticas de cor de uma imagem. Nenhuma envolve aprendizado."""
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # Excess Green (Woebbecke et al.), o indice de vegetacao RGB classico
    exg = 2.0 * g - r - b
    # a* do CIELAB: quanto MAIS NEGATIVO, mais verde
    lab_a = rgb2lab(rgb)[..., 1]
    # razao verde simples, robusta a iluminacao
    green_ratio = g / (r + g + b + 1e-6)

    return np.array([
        exg.mean(),
        exg.std(),
        float((exg > 0.05).mean()),      # fracao de pixels "verdes"
        lab_a.mean(),
        float((lab_a < -5.0).mean()),    # fracao de pixels verdes no CIELAB
        green_ratio.mean(),
    ], dtype=np.float64)


def load_split(split):
    """Devolve (paths, X, y) com y=1 para NOK."""
    files = list_class_files(DATA / split)
    paths, X, y = [], [], []
    for cls, ps in files.items():
        for p in ps:
            paths.append(p)
            X.append(extract(p))
            y.append(CLASSES.index(cls))
    return paths, np.vstack(X), np.array(y)


def best_threshold(scores, y):
    """Limiar que maximiza o macro-F1 no conjunto em que e ajustado."""
    best_t, best_f1 = None, -1.0
    for t in np.unique(scores):
        f1 = metrics_table(y, (scores >= t).astype(int))["macro_f1"]
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def run_bootstrap(feat_idx, sign, Xtr_by_cls, Xte, yte):
    """30 replicas de um classificador de limiar sobre UMA feature."""
    rows = []
    for i in range(N_REP):
        res = bootstrap_resample(Xtr_by_cls, seed=SEED_BASE + i)
        s = np.concatenate([np.array([v[feat_idx] for v in vs]) for vs in res.values()])
        yb = np.concatenate([np.full(len(vs), CLASSES.index(c)) for c, vs in res.items()])
        t = best_threshold(sign * s, yb)
        rows.append(metrics_table(yte, (sign * Xte[:, feat_idx] >= t).astype(int)))
    return rows


def run_bootstrap_logreg(Xtr_by_cls, Xte, yte):
    """30 replicas de regressao logistica sobre as 6 features."""
    rows = []
    for i in range(N_REP):
        res = bootstrap_resample(Xtr_by_cls, seed=SEED_BASE + i)
        Xb = np.vstack([np.vstack(vs) for vs in res.values()])
        yb = np.concatenate([np.full(len(vs), CLASSES.index(c)) for c, vs in res.items()])
        sc = StandardScaler().fit(Xb)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xb), yb)
        rows.append(metrics_table(yte, clf.predict(sc.transform(Xte))))
    return rows


def resumo(nome, rows):
    f1 = np.array([r["macro_f1"] for r in rows])
    nok_r = np.array([r["NOK_recall"] for r in rows])
    print(f"  {nome:<22} macro-F1 {f1.mean():5.2f} +- {f1.std(ddof=1):4.2f}"
          f"   NOK recall {nok_r.mean():5.2f} +- {nok_r.std(ddof=1):4.2f}")
    return dict(name=nome, macro_f1_mean=float(f1.mean()), macro_f1_std=float(f1.std(ddof=1)),
                nok_recall_mean=float(nok_r.mean()), nok_recall_std=float(nok_r.std(ddof=1)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("extraindo features de cor (sem GPU, sem treino)...")
    tr_paths, Xtr, ytr = load_split("train")
    te_paths, Xte, yte = load_split("test")
    print(f"  treino {len(ytr)} imagens, teste {len(yte)} imagens")

    # agrupa o treino por classe, no formato que bootstrap_resample espera
    Xtr_by_cls = {c: [Xtr[j] for j in range(len(ytr)) if ytr[j] == CLASSES.index(c)]
                  for c in CLASSES}

    print("\n=== bootstrap, 30 replicas (protocolo identico ao das CNNs) ===")
    resumos = []
    for name, sign in SINGLE:
        idx = FEATURES.index(name)
        resumos.append(resumo(f"{name} (limiar)", run_bootstrap(idx, sign, Xtr_by_cls, Xte, yte)))
    rows_lr = run_bootstrap_logreg(Xtr_by_cls, Xte, yte)
    resumos.append(resumo("logreg 6 features", rows_lr))
    # metricas por classe de cada replica -- coluna "Colour" da Tabela II do artigo
    with open(OUT / "logreg_bootstrap.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["replicate"] + list(rows_lr[0].keys()))
        w.writeheader()
        w.writerows({"replicate": i, **r} for i, r in enumerate(rows_lr))

    # --- ranking no teste: AP e ROC-AUC, comparaveis diretamente com as CNNs ---
    print("\n=== qualidade de ranqueamento no teste (ajuste unico no treino completo) ===")
    rank = []
    for name, sign in SINGLE:
        s = sign * Xte[:, FEATURES.index(name)]
        ap, auc = average_precision_score(yte, s), roc_auc_score(yte, s)
        print(f"  {name:<22} AP {ap:.3f}   ROC-AUC {auc:.3f}")
        rank.append(dict(name=name, ap=float(ap), roc_auc=float(auc)))

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    s_lr = clf.predict_proba(sc.transform(Xte))[:, 1]
    ap, auc = average_precision_score(yte, s_lr), roc_auc_score(yte, s_lr)
    print(f"  {'logreg 6 features':<22} AP {ap:.3f}   ROC-AUC {auc:.3f}")
    rank.append(dict(name="logreg", ap=float(ap), roc_auc=float(auc)))
    print(f"  {'--- CNNs (referencia)':<22} ResNet18 AP 0.842 / AUC 0.935"
          f"   YOLO26 AP 0.863 / AUC 0.943")

    # --- as CNNs estao seguindo o verde? correlacao entre os escores ---
    npz = ROOT / "results" / "nok_analysis" / "scores_test.npz"
    corr = {}
    if npz.exists():
        d = np.load(npz, allow_pickle=True)
        order = {Path(str(p)).name: k for k, p in enumerate(d["paths"])}
        perm = [order[p.name] for p in te_paths]
        print("\n=== as CNNs seguem o verde global? (correlacao de Pearson no teste) ===")
        for cnn in ("resnet", "yolo"):
            s_cnn = d[cnn][perm]
            for name, sign in SINGLE:
                c = float(np.corrcoef(sign * Xte[:, FEATURES.index(name)], s_cnn)[0, 1])
                corr[f"{cnn}_vs_{name}"] = c
            c_lr = float(np.corrcoef(s_lr, s_cnn)[0, 1])
            corr[f"{cnn}_vs_logreg"] = c_lr
            print(f"  {cnn:<8} exg_mean {corr[f'{cnn}_vs_exg_mean']:+.3f}"
                  f"   lab_a_mean {corr[f'{cnn}_vs_lab_a_mean']:+.3f}"
                  f"   logreg {c_lr:+.3f}")

    with open(OUT / "summary.json", "w") as f:
        json.dump(dict(bootstrap=resumos, ranking=rank, correlations=corr), f, indent=2)
    np.savez(OUT / "features_test.npz", paths=[str(p) for p in te_paths], X=Xte, y=yte,
             features=FEATURES, logreg_score=s_lr)
    print(f"\nOK. Resultados em {OUT}/")


if __name__ == "__main__":
    main()

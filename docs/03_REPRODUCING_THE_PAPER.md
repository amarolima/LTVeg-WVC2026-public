# 03 — Reproducing the paper

| Level | What is recomputed | Needs | Time |
|---|---|---|---|
| **0** | All tables, Fig. 3, statistics, threshold sweep, geography, checklist, from `results/` | nothing | ~1 min, CPU |
| **1** | Level 0 + Figs. 2 and 4 + colour baseline from the pixels | regenerated images ([02](02_REGENERATING_THE_DATASET.md)) | +20 s |
| **2** | Every training run: 30 + 30 bootstraps, ablation, single runs, cost | images + NVIDIA GPU | ~4–6 h |

Levels 0 and 1 are the notebook (`notebooks/reproduce_paper.ipynb`). It detects whether the 351
images are present in `data/dataset/`. Its outputs go to `results/paper/`, and its final checklist
(`results/paper/checklist_paper_numbers.csv`) compares **125 numbers printed in the paper** with the
recomputed values.

## Paper → code → stored result

| In the paper | Produced by | Stored result |
|---|---|---|
| Fig. 1, Alg. 1, Sec. III-A–C | `src/extraction/extrator_imagens.py` (QGIS) | `data/metadata/`, `data/labels.csv` |
| Table I | sampling in `extrator_imagens.py` | `data/metadata/coordenadas_sorteadas/` |
| Sec. III-D: 170 lines, latitudes | `src/analysis/analise_geografica.py` | — (notebook §1.5) |
| Fig. 2 | dataset images | — (needs the images) |
| Sec. IV: AdamW for YOLO26 | `src/training/yolo_ablation.py`, `bootstrap_yolo.py --config …` | `results/yolo_ablation/`, `results/bootstrap_yolo_{auto,nano}/` |
| Sec. V-A, Table II (ResNet18, YOLO26) | `bootstrap_resnet.py`, `bootstrap_yolo.py --config explicit_adamw` | `results/bootstrap_{resnet,yolo_adamw}/results.csv` |
| Table II (Colour), colour ROC-AUC | `src/training/color_baseline.py` | `results/color_baseline/` |
| Welch test, CI of the difference | `src/analysis/bootstrap_compare.py` + notebook §3.4 | `results/bootstrap_comparison.json` |
| Sec. V-C: AP, ROC-AUC, sweep | `src/analysis/nok_analysis.py` | `results/nok_analysis/` |
| Fig. 3, Fig. 4 | `src/figures/refazer_figuras_wvc.py` | built from `results/nok_analysis/scores_test.npz` |
| Table III | `src/training/compute_cost.py` | `results/compute_cost.json` |
| Sec. V-E: leakage, overlap | `analise_geografica.py`, `analise_sobreposicao.py` | — (notebook §1.5–1.6) |

**Which models produce which numbers.** Table II and Sections V-A/V-B are averages over the **30
bootstrap replicates**; those runs stored only their metrics, not weights or per-image scores. Section
V-C, Figs. 3 and 4 and the threshold sweep come from **one representative run per model**:
`train_baseline.py` (ResNet18, seed 42) and `yolo_ablation.py` config `explicit_adamw` (YOLO26,
seed 42). Their per-image test scores are in `results/nok_analysis/scores_test.npz`. The trained
weights are not distributed, because they were trained on the Google imagery. Retrain them (below)
to recompute the scores.

## Level 2 — retraining

With the images in `data/dataset/` and an NVIDIA GPU:

```bash
bash scripts/retrain_all.sh                 # everything (~4–6 h on an RTX 4050)
bash scripts/retrain_all.sh resnet yolo     # selected stages: single resnet ablation yolo yolo_auto yolo_nano cost compare
```

Outputs go to `results/rerun/`; the original results are never overwritten. Bootstrap runs are
resumable. To recompute the Sec. V-C scores from your own single runs:

```bash
mkdir -p checkpoints
cp results/rerun/train_baseline/resnet18_best.pt checkpoints/resnet18_best.pt
cp results/rerun/yolo_ablation/runs/explicit_adamw/weights/best.pt checkpoints/yolo26s_adamw_best.pt
python src/analysis/nok_analysis.py          # overwrites results/nok_analysis/ — commit nothing from it
```

**Expect close, not identical, numbers.** cuDNN and parallel data loading are nondeterministic even
with fixed seeds. The mean of 30 replicates should fall within about 1 point of the paper, while a
single run can move more. Latency and epoch time depend on the hardware. With `SAME_SCENE`/`CHECK`
images (doc 02) the data itself differs slightly.

## Conventions worth knowing

* Class order is **OK = 0, NOK = 1** (NOK is the positive class). `ImageFolder` and Ultralytics sort
  folders alphabetically (NOK = 0), and every script remaps explicitly through `model.names`.
* Standard deviations over replicates use `ddof=1`.
* Scripts are run from the repository root (`python src/.../script.py`).

# results/ — original outputs of the experiments

These are the files produced for the paper; the notebook reads them. Retraining writes to
`results/rerun/` (`scripts/retrain_all.sh`) and never overwrites them. No file here contains image pixels.

| File | Produced by | Content | Used in |
|---|---|---|---|
| `bootstrap_resnet/results.csv` | `bootstrap_resnet.py` (seeds 1000–1029) | 30 ResNet18 replicates: `val_macro_f1` + 7 test metrics | Table II, V-A, Welch |
| `bootstrap_yolo_adamw/results.csv` | `bootstrap_yolo.py --config explicit_adamw` (2000–2029) | 30 YOLO26s-cls replicates, AdamW lr 1e-4, cosine — **the paper's model** | Table II, V-A, Welch |
| `bootstrap_yolo_auto/results.csv` | `--config baseline_current` | YOLO26 with `optimizer=auto` | ablation (Sec. IV) |
| `bootstrap_yolo_nano/results.csv` | `--config light_aug_adamw_nano` | YOLO26n, light augmentation, AdamW | ablation |
| `*/summary.json` | `bootstrap_common.summarize` | mean, std (ddof=1), percentile CI | quick look |
| `yolo_ablation/results.csv` | `yolo_ablation.py` (seed 42, single runs) | 7 YOLO26 configurations | ablation |
| `yolo_ablation/explicit_adamw_{epochs.csv,args.yaml}` | same | per-epoch log and arguments of the representative YOLO26 run | — |
| `train_baseline/{results,history}.json` | `train_baseline.py` (seed 42) | representative ResNet18 run | — |
| `nok_analysis/scores_test.npz` | `nok_analysis.py` | `paths`, `y` (1 = NOK), P(NOK) of `resnet` and `yolo` on the 69 test images | Figs. 3–4, V-C, V-E |
| `nok_analysis/sweep_{resnet18,yolo26}.csv`, `summary.json` | same | threshold sweep; AP, ROC-AUC | V-C |
| `color_baseline/logreg_bootstrap.csv` | `color_baseline.py` (3000–3029) | 30 colour-baseline replicates, per class | Table II (Colour) |
| `color_baseline/summary.json`, `features_test.npz` | same | single-index variants, AP/ROC-AUC, correlations; 6 colour statistics of the test images | V-B, V-C |
| `compute_cost.json` | `compute_cost.py` | parameters, GFLOPs, checkpoint size, latency, epoch time | Table III |
| `bootstrap_comparison.json` | `bootstrap_compare.py` | Welch test per metric, ResNet18 vs YOLO26 AdamW | V-B |
| `environment_freeze.txt` | `pip freeze` | environment used for training | — |

Hardware: NVIDIA RTX 4050 Laptop GPU (6 GB), Ultralytics 8.4.87, PyTorch 2.6.0 + CUDA 12.4.

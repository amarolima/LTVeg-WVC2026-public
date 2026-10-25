# Vegetation Encroachment Screening in Transmission-Line Rights-of-Way from Satellite Imagery

Code, labels and results needed to reproduce every table, figure and number of the paper

> D. C. S. Braga, J. N. Gois, W. L. Passos, A. A. Lima, G. M. Araujo.
> *Vegetation Encroachment Screening in Transmission-Line Rights-of-Way from Satellite Imagery.*
> Workshop de Visão Computacional (WVC), 2026. PPEEL/CEFET-RJ, Rio de Janeiro, Brazil.

The paper classifies 650×650 satellite images centred on towers of the Brazilian transmission grid
as **OK** (vegetation outside the right-of-way) or **NOK** (vegetation encroaching on it). It
fine-tunes YOLO26 and ResNet18, retrains each 30 times by bootstrap, compares them with a colour
baseline and analyses the operating point for field-inspection screening.

| | test macro-F1 (30 replicates) | ROC-AUC | parameters | latency |
|---|---|---|---|---|
| Colour baseline (logistic regression, 6 features) | 64.7 % ± 2.5 | 0.772 | — | — |
| ResNet18 | 83.1 % ± 4.2 | 0.935 | 11.18 M | 8.57 ms |
| **YOLO26s-cls (AdamW)** | 81.8 % ± 5.0 | 0.943 | **5.45 M** | **4.46 ms** |

## The images are not in this repository

The 351 images are crops of **Google Maps** satellite imagery, whose terms do not allow bulk
redistribution. Instead, this repository provides:

* **`data/labels.csv`**: for each of the 351 images, the file name, the **expert label**, the
  **train/val/test split** of the paper, the tower index, voltage, transmission line, coordinates and
  **four fingerprints** (SHA-256 of the file and of the pixels, plus two perceptual hashes);
* **the code to regenerate the images** in QGIS for exactly the same towers
  (`src/extraction/regenerate_images_qgis.py`);
* **a verifier** (`src/dataset/verify_dataset.py`) that tells whether each regenerated image is
  identical, the same scene, or different from the one used in the paper, and arranges the accepted
  images into the folder structure used by the training code.

```bash
conda create -n qgis -c conda-forge qgis=3.40.3                       # once
conda run -n qgis python src/extraction/regenerate_images_qgis.py --out regenerated/    # ~2 min, headless
python src/dataset/verify_dataset.py --images regenerated/ --organize data/dataset
```

Google has updated its imagery since the dataset was built, so bit-identical copies are no longer
possible. On 25/10/2026, 305 of the 351 regenerated images were the same scene, 27 needed a visual
check and 19 showed a newer acquisition. See [docs/02_REGENERATING_THE_DATASET.md](docs/02_REGENERATING_THE_DATASET.md).

## Quick start (no GPU, no images, ~1 minute)

```bash
conda env create -f environment.yml && conda activate ltveg
jupyter nbconvert --to notebook --execute --inplace notebooks/reproduce_paper.ipynb
```

The notebook regenerates Tables I–III, Fig. 3, the statistical tests, the threshold sweep and the
geographic analyses from the stored experiment results. It ends with a **checklist of 125 numbers
printed in the paper**, all of which must match. Figs. 2 and 4 and the colour baseline are also
recomputed once the images have been regenerated.

## Repository layout

```text
data/
  labels.csv                  labels, split, coordinates and fingerprints of the 351 images
  metadata/                   EPE tower spreadsheet (194,299 towers) and the 4×100 sampled towers
  dataset/                    empty; filled by verify_dataset.py --organize
notebooks/reproduce_paper.ipynb   every table, figure and number of the paper + checklist
src/
  extraction/                 original extractor and regeneration script (QGIS)
  dataset/                    fingerprints, labels.csv builder, verifier
  training/                   ResNet18, YOLO26, bootstrap, ablation, colour baseline, cost
  analysis/                   Welch test, NOK threshold analysis, geographic and overlap analyses
  figures/                    figures at their final size in the paper
results/                      original outputs of every experiment (see results/README.md)
scripts/retrain_all.sh        reruns every training (hours of GPU)
docs/                         installation, dataset regeneration, reproduction, known issues
```

## Documentation

| | |
|---|---|
| [01 — Installation](docs/01_INSTALLATION.md) | environment and versions |
| [02 — Regenerating the dataset](docs/02_REGENERATING_THE_DATASET.md) | QGIS, fingerprints, verification |
| [03 — Reproducing the paper](docs/03_REPRODUCING_THE_PAPER.md) | paper → code → result map; retraining |
| [04 — Known issues](docs/04_KNOWN_ISSUES.md) | inconsistencies found in the paper and limitations |

## Citation

See [`CITATION.cff`](CITATION.cff). Figures that show dataset images must credit *Imagery © Google*.

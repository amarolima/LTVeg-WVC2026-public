# 01 — Installation

```bash
conda env create -f environment.yml      # creates the "ltveg" environment
conda activate ltveg
```

The environment installs PyTorch with CUDA 12.4. It also works on a machine without an NVIDIA GPU,
in which case it runs on CPU. **The notebook needs no GPU.**

Without conda:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0   # or .../whl/cpu
pip install -r requirements.txt
```

Check the installation:

```bash
python -c "import torch, ultralytics; print(torch.__version__, ultralytics.__version__, torch.cuda.is_available())"
jupyter nbconvert --to notebook --execute --inplace notebooks/reproduce_paper.ipynb   # must end with 125/125
```

## Versions that matter

| Package | Version | Why |
|---|---|---|
| `ultralytics` | 8.4.87 | YOLO26; augmentation and optimizer defaults change between releases |
| `torch` / `torchvision` | 2.6.0 / 0.21.0 | ResNet18 ImageNet weights |
| `scikit-learn`, `scikit-image` | 1.9.0, 0.26.0 | metrics, colour baseline |
| `ultralytics-thop` | 2.0.20 | FLOP count (Table III) |

The full `pip freeze` of the environment used for the paper is in `results/environment_freeze.txt`.
The experiments ran on an NVIDIA RTX 4050 Laptop GPU (6 GB) with CUDA 12.4.

QGIS 3.x is needed only to regenerate the images ([02](02_REGENERATING_THE_DATASET.md)). It is used
through its own Python console, not through this environment.

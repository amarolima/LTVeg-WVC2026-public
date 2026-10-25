# 02 — Regenerating the dataset

The images cannot be redistributed (Google Maps terms), but every image can be regenerated for the
**same tower, at the same scale and with the same rendering**, and checked against the original.

## 1. What `data/labels.csv` contains

| Column | Content |
|---|---|
| `file` | `<tower_index>_<group>_<voltage>.png`, e.g. `44620_G1_800.png` |
| `split` | `train` / `val` / `test`: the split of the paper (210 / 72 / 69). **Do not change it** |
| `label` | `OK` (vegetation clearly outside the ROW) / `NOK` (vegetation clearly inside it) |
| `tower_index` | row (0-based) of `data/metadata/coordenadas_extraidas.xlsx` |
| `group`, `voltage_kv` | voltage group (G1–G4, Table I) and line voltage |
| `line_name`, `lon`, `lat` | transmission line and tower coordinate, EPSG:4674 (from the EPE spreadsheet) |
| `sha256_file` | SHA-256 of the original PNG file |
| `sha256_pixels` | SHA-256 of the decoded RGB pixels (the alpha channel is constant and dropped) |
| `phash`, `dhash` | 64-bit perceptual hashes (DCT hash and difference hash), in hex |

The fingerprints are computed by `src/dataset/fingerprints.py`, and `labels.csv` was produced by
`src/dataset/make_labels.py` from the original images.

**Labelling criterion (qualitative, single expert):** vegetation located clearly underneath or very
close to the transmission lines → NOK; images in which the ROW is clearly demarcated and free of
vegetation → OK. The label refers to the image as acquired at extraction time (May 2026).

## 2. Regenerating the images in QGIS

Requirements: QGIS 3.40 (the LTR series used to build the dataset) and internet access for the
Google Satellite XYZ layer. No display is needed. The simplest route is the conda-forge build:

```bash
conda create -n qgis -c conda-forge qgis=3.40.3          # ~1-2 GB, independent of the "ltveg" env
conda run -n qgis python src/extraction/regenerate_images_qgis.py --labels data/labels.csv --out regenerated/
```

The script renders the 351 towers of `labels.csv`, with no random sampling, using the parameters of
the original extractor (`src/extraction/extrator_imagens.py`):

* EPSG:4674 → EPSG:3857 (Web Mercator, the CRS of the Google mosaic);
* extent of 162.5 m × 162.5 m centred on the tower;
* 650 × 650 px, i.e. 0.25 m/pixel nominal (the layer serves zoom 19, which QGIS resamples);
* only the Google Satellite layer is rendered.

It takes about 2 minutes, is resumable (images already in the output folder are skipped) and is
deterministic: two runs give identical pixels. It also runs from the Python console of a desktop
QGIS: set `LABELS_CSV` and `OUTPUT_DIR` at the top of the file and press *Run*.

## 3. Verifying and organizing

```bash
python src/dataset/verify_dataset.py --images <OUTPUT_DIR> --organize data/dataset
```

Each image is compared with the fingerprints of the image of the same name and classified as:

| Status | Rule | Meaning |
|---|---|---|
| `IDENTICAL_FILE` | same `sha256_file` | the very same file |
| `IDENTICAL_PIXELS` | same `sha256_pixels` | same pixels, different PNG encoding (e.g. another QGIS version) |
| `SAME_SCENE` | pHash ≤ 10 **and** dHash ≤ 10 | same place; tiny rendering, compression or brightness differences |
| `CHECK` | pHash or dHash 11–17 | probably the same place, shifted by several pixels, or a different acquisition; inspect visually |
| `DIFFERENT` | otherwise | most likely a newer Google acquisition (or a wrong tower) |
| `MISSING` | — | not regenerated |

A per-image report is written to `results/dataset_verification.csv`. With `--organize`, images up to
`SAME_SCENE` are copied into `data/dataset/{train,val,test}/{OK,NOK}/`, the layout every script
expects. Use `--accept CHECK` or `--accept ALL` to be more permissive.

### How the thresholds were chosen

The hashes were calibrated on the 351 original images (Hamming distance, out of 64 bits):

| | pHash | dHash |
|---|---|---|
| **Different images** (61,425 pairs) | min 14 · 1st percentile 22 · median 32 | min 13 · 1st percentile 22 · median 32 |
| Same image, re-encoded PNG | 0 | 0 |
| Same image, JPEG q70–q90 | ≤ 2 | ≤ 4 |
| Same image, brightness +10 % | ≤ 4 | ≤ 9 |
| Same image, shifted 2 px / 5 px (0.5 / 1.25 m) | ≤ 8 / ≤ 12 | ≤ 9 / ≤ 14 |
| Same image, scaled +3 % | ≤ 12 | ≤ 12 |
| Same image, shifted 10 px / 20 px | ≤ 18 / ≤ 30 | ≤ 20 / ≤ 26 |

The only pair of *different* files closer than 13 is `57854_G4_230.png` × `57896_G4_230.png`
(distance 2/4). These are two towers 2 m apart whose footprints overlap by 98 %, so they really are
the same scene.

## 4. What to expect

**Exact copies are no longer possible.** The original images were extracted in May 2026. Since then
Google has re-encoded or replaced the tiles. We regenerated all 351 images on 25/10/2026, and
compared them with the fingerprints:

| Status | Images |
|---|---|
| `IDENTICAL_FILE` / `IDENTICAL_PIXELS` | 0 |
| `SAME_SCENE` | 305 (87 %) |
| `CHECK` | 27 |
| `DIFFERENT` | 19 |

Over all 351 images the mean absolute error per pixel is 3.5 grey levels out of 255 (median). An
independent re-implementation of the rendering (zoom-19 tiles mosaicked and bilinearly resampled in
pure Python) gives the **same** error, within 0.02 levels, and differs from the QGIS output by
0.03 levels on average. So the residual is in the imagery, not in the rendering, and the geometry
is exact: the best sub-pixel offset is zero. Visual inspection of `DIFFERENT` images shows the same
place, aligned, but with a newer acquisition (season, colour and shadows changed).

How much this matters: recomputing the colour baseline on the 351 images regenerated that day gives
a test macro-F1 of 62.3 % instead of 64.7 %, and ROC-AUC 0.735 instead of 0.772.

| Your images | What you get |
|---|---|
| All `IDENTICAL_*` | The original dataset. Retraining should reproduce the paper within run-to-run GPU nondeterminism (doc 03), and the colour baseline exactly |
| Some `SAME_SCENE` | Practically the same data. Small numerical differences when retraining |
| `CHECK` / `DIFFERENT` | Newer imagery for those towers. **The labels may no longer hold**, because vegetation changes and the ROW may have been cleared or overgrown. Exclude them, or re-label them, before drawing conclusions |

In every case, the notebook reproduces all tables and figures of the paper from the stored results
in `results/`, which do not depend on your images.

## 5. Terms of use

Regenerated images are Google imagery: use them under the Google Maps/Google Earth terms that apply
to you, do not redistribute them, and credit *Imagery © Google* in figures. The EPE data in
`data/metadata/` is public data from the Brazilian Energy Research Company (WEBMAP,
https://gisepeprd2.epe.gov.br/WebMapEPE/). Two columns with the editor's login were removed from it;
the row order, which the image names depend on, is unchanged.

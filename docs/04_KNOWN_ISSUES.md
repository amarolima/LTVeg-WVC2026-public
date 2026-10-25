# 04 — Known issues

## Inconsistencies found in the published paper

Neither of them changes a conclusion of the paper.

1. **Table II mixes two standard deviations (≤ 0.2 point).** The per-class rows of ResNet18 and
   YOLO26 were printed with the population std (`ddof=0`), whereas their macro-F1 row and the whole
   Colour column use the sample std (`ddof=1`). Example: NOK recall of ResNet18 is printed as 11.1
   and is 11.3 with `ddof=1`. The notebook writes both versions (`results/paper/table2_test*.csv`).
2. **Group G4 contains no 138 kV tower.** Table I defines G4 as 230 and 138 kV, but the Basic Network
   spreadsheet has no 138 kV vertex, so all 100 sampled G4 towers are 230 kV. Towers at 525 kV
   (12,624) and 132 kV (33) belong to no group and were never sampled.

## Limitations (as stated in the paper)

* Single image source (Google Maps), one acquisition per location, unknown dates.
* A single expert annotator, with a qualitative criterion and no written protocol. One test image
  (`179391_G4_230.png`, labelled OK) is predicted NOK with ~1.00 / 0.96 confidence by both models and
  may be a label error.
* Small, imbalanced dataset: 18 NOK test images, unevenly spread across groups (9 in G3, 2 in G1).
* The bootstrap resamples only the training set; the reported dispersion is sensitivity to the
  training composition, not uncertainty over the test set.
* Random per-image split. Leakage was measured *a posteriori* and is small: 25 % of test images share
  a line with training, the median distance to the nearest training image is 30.5 km, and one
  overlapping pair crosses train/test. A geographically grouped split is future work. Grouping by
  line name is not enough: the overlapping pair belongs to circuits C1 and C2 of the same corridor,
  which have different names.
* The YOLO26 optimizer was screened on test performance before the final bootstrap.
* The threshold analysis (Sec. V-C) comes from one run per model, because the bootstrap runs did not
  store per-image scores.

## Reproducibility caveats

* `src/extraction/extrator_imagens.py` samples towers without a seed (`DataFrame.sample(n)`). The
  towers actually used are those listed in `data/labels.csv`; regenerate them with
  `regenerate_images_qgis.py`.
* The curation step (49 of 400 images discarded) and the train/val/test split were done by hand, and
  their procedure was not recorded. The split is fixed by `labels.csv`.
* `regenerate_images_qgis.py` has not yet been run end to end by the authors.

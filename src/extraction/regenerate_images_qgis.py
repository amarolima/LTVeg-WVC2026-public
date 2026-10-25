"""
Regenerates the 351 dataset images with QGIS, for exactly the towers listed in
data/labels.csv (no random sampling).

Rendering follows src/extraction/extrator_imagens.py, the script that produced the
original dataset (QGIS 3.40 LTR): Google Satellite XYZ layer (zmax 19), EPSG:4674 ->
EPSG:3857, a 162.5 m x 162.5 m extent centred on the tower (650 px x 0.25 m/px),
rendered at 650 x 650 px. Only the Google layer is rendered.

Runs headless, from the command line, with the Python of a QGIS installation, e.g.
the conda environment of docs/01_INSTALLATION.md:

    conda run -n qgis python src/extraction/regenerate_images_qgis.py \
        --labels data/labels.csv --out regenerated/
    python src/dataset/verify_dataset.py --images regenerated/ --organize data/dataset

It also runs inside the QGIS Python console (Plugins > Python Console > Show Editor):
set LABELS_CSV and OUTPUT_DIR below and press Run. The script is resumable: images
already present in the output folder are skipped.

Tested with QGIS 3.40.3 (conda-forge) on 25/10/2026: all 351 images rendered; output DPI
does not change the result.
"""
import csv
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")      # no display needed

from qgis.core import (QgsApplication, QgsCoordinateReferenceSystem, QgsCoordinateTransform,
                       QgsMapRendererParallelJob, QgsMapSettings, QgsPointXY, QgsProject,
                       QgsRasterLayer, QgsRectangle)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor

# used only when the script is run from the QGIS Python console
LABELS_CSV = r"/path/to/repo/data/labels.csv"
OUTPUT_DIR = r"/path/to/regenerated_images"

IMAGE_PX = 650          # extrator_imagens.py: image_width = image_height = 650
RESOLUTION = 0.25       # extrator_imagens.py: resolution = 0.25 (m/pixel)
GOOGLE_URL = ("type=xyz&url=https://mt1.google.com/vt/lyrs%3Ds%26x%3D%7Bx%7D%26y%3D%7By%7D"
              "%26z%3D%7Bz%7D&zmax=19&zmin=0")


def regenerate(labels_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    layer = QgsRasterLayer(GOOGLE_URL, "Google Satellite", "wms")
    if not layer.isValid():
        raise RuntimeError("could not load the Google Satellite layer (internet access?)")
    QgsProject.instance().addMapLayer(layer)

    src_crs = QgsCoordinateReferenceSystem("EPSG:4674")     # EPE shapefile
    dst_crs = QgsCoordinateReferenceSystem("EPSG:3857")     # Google mosaic
    transform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
    half = IMAGE_PX * RESOLUTION / 2                          # 81.25 m

    with open(labels_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for k, r in enumerate(rows, 1):
        out = os.path.join(output_dir, r["file"])
        if os.path.exists(out):
            continue
        p = transform.transform(QgsPointXY(float(r["lon"]), float(r["lat"])))
        settings = QgsMapSettings()
        settings.setDestinationCrs(dst_crs)
        settings.setLayers([layer])
        settings.setBackgroundColor(QColor(255, 255, 255))
        settings.setOutputSize(QSize(IMAGE_PX, IMAGE_PX))
        settings.setExtent(QgsRectangle(p.x() - half, p.y() - half, p.x() + half, p.y() + half))
        job = QgsMapRendererParallelJob(settings)
        job.start()
        job.waitForFinished()
        if not job.renderedImage().save(out):
            raise RuntimeError(f"could not write {out}")
        if k % 25 == 0 or k == len(rows):
            print(f"{k}/{len(rows)}")
    print(f"done: {len(rows)} images in {output_dir}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Regenerate the dataset images with QGIS (headless).")
    ap.add_argument("--labels", default="data/labels.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    app = QgsApplication([], False)
    app.initQgis()
    try:
        regenerate(args.labels, args.out)
    finally:
        app.exitQgis()


def _in_qgis_console():
    utils = sys.modules.get("qgis.utils")
    return utils is not None and getattr(utils, "iface", None) is not None


if _in_qgis_console():
    regenerate(LABELS_CSV, OUTPUT_DIR)
elif __name__ == "__main__":
    main()

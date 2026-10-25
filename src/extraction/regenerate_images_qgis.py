"""
Regenerates the 351 dataset images inside QGIS, for exactly the towers listed in
data/labels.csv (no random sampling).

Rendering follows src/extraction/extrator_imagens.py, the script that produced the
original dataset: Google Satellite XYZ layer, EPSG:4674 -> EPSG:3857, a 162.5 m x
162.5 m extent centred on the tower (650 px x 0.25 m/px), rendered at 650 x 650 px.
The only deliberate difference is that ONLY the Google layer is rendered, so that
other layers of your project cannot leak into the image.

!!! Run it in the QGIS Python console (Plugins > Python Console > Show Editor >
!!! open this file > Run). It does not run in a plain Python interpreter.

Before running, set the two paths below. Afterwards, from the repository root:
    python src/dataset/verify_dataset.py --images <OUTPUT_DIR> --organize data/dataset

NOTE: this script was written for the public release and could not be executed by
the authors' tooling at release time (no QGIS available); the rendering calls are the
same as the original extractor's. Please report problems.
"""
import csv
import os

from PyQt5.QtCore import QSize
from qgis.core import (QgsCoordinateReferenceSystem, QgsCoordinateTransform,
                       QgsMapRendererParallelJob, QgsPointXY, QgsProject,
                       QgsRasterLayer, QgsRectangle)
from qgis.utils import iface

# ========== SET THESE ==========
LABELS_CSV = r"/path/to/repo/data/labels.csv"
OUTPUT_DIR = r"/path/to/regenerated_images"
# ===============================

IMAGE_PX = 650          # extrator_imagens.py: image_width = image_height = 650
RESOLUTION = 0.25       # extrator_imagens.py: resolution = 0.25 (m/pixel)
CAMERA_SCALE = 50       # extrator_imagens.py: altura_camera = 50 (canvas.zoomScale)
GOOGLE_URL = ("type=xyz&url=https://mt1.google.com/vt/lyrs%3Ds%26x%3D%7Bx%7D%26y%3D%7By%7D"
              "%26z%3D%7Bz%7D&zmax=19&zmin=0")


def google_layer():
    layer = QgsRasterLayer(GOOGLE_URL, "Google Satellite", "wms")
    if not layer.isValid():
        raise RuntimeError("could not load the Google Satellite layer")
    QgsProject.instance().addMapLayer(layer)
    return layer


def render(layer, extent, crs, out_path):
    canvas = iface.mapCanvas()
    canvas.setDestinationCrs(crs)
    canvas.setExtent(extent)
    canvas.zoomScale(CAMERA_SCALE)
    canvas.refresh()
    canvas.waitWhileRendering()

    settings = canvas.mapSettings()
    settings.setLayers([layer])
    settings.setOutputSize(QSize(IMAGE_PX, IMAGE_PX))
    settings.setExtent(extent)
    job = QgsMapRendererParallelJob(settings)
    job.start()
    job.waitForFinished()
    job.renderedImage().save(out_path)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    src_crs = QgsCoordinateReferenceSystem("EPSG:4674")   # EPE shapefile
    dst_crs = QgsCoordinateReferenceSystem("EPSG:3857")   # Google mosaic
    transform = QgsCoordinateTransform(src_crs, dst_crs, QgsProject.instance())
    half = IMAGE_PX * RESOLUTION / 2                        # 81.25 m

    layer = google_layer()
    with open(LABELS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for k, r in enumerate(rows, 1):
        out = os.path.join(OUTPUT_DIR, r["file"])
        if os.path.exists(out):
            continue                                        # resumable
        p = transform.transform(QgsPointXY(float(r["lon"]), float(r["lat"])))
        extent = QgsRectangle(p.x() - half, p.y() - half, p.x() + half, p.y() + half)
        render(layer, extent, dst_crs, out)
        print(f"{k}/{len(rows)} {r['file']}")
    print(f"done: {len(rows)} images in {OUTPUT_DIR}")


main()

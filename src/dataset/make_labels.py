"""
Builds data/labels.csv from the ORIGINAL dataset (data/dataset/{train,val,test}/{OK,NOK}).

This is the script that produced the labels.csv shipped with this repository. It needs
the original images, which are not distributed (Google Maps terms), so a public user
does not need to run it -- it is here so that the provenance of every column is clear.

Columns:
    file           image file name, <tower_index>_<group>_<voltage>.png
    split          train / val / test (the split used in the paper)
    label          OK / NOK (expert annotation)
    tower_index    row of data/metadata/coordenadas_extraidas.xlsx (0-based)
    group, voltage_kv
    line_name, lon, lat   from the EPE tower spreadsheet (EPSG:4674)
    sha256_file, sha256_pixels, phash, dhash   fingerprints, see fingerprints.py

Usage (from the repository root, with the original images in data/dataset/):
    python src/dataset/make_labels.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fingerprints import fingerprint

DATA = Path("data/dataset")
TOWERS = Path("data/metadata/coordenadas_extraidas.xlsx")
OUT = Path("data/labels.csv")


def main():
    towers = pd.read_excel(TOWERS)
    rows = []
    for p in sorted(DATA.glob("*/*/*.png")):
        idx, group, voltage = p.stem.split("_")
        t = towers.iloc[int(idx)]
        assert int(t.Tensao) == int(voltage), f"{p.name}: voltage differs from the spreadsheet"
        rows.append(dict(file=p.name, split=p.parts[-3], label=p.parts[-2], tower_index=int(idx),
                         group=group, voltage_kv=int(voltage), line_name=t.Nome,
                         lon=t.X_coord_pre, lat=t.Y_coord_pre, **fingerprint(p)))
    df = pd.DataFrame(rows)
    order = {"train": 0, "val": 1, "test": 2}
    df = df.sort_values(["split", "label", "file"], key=lambda c: c.map(order) if c.name == "split" else c)
    df.to_csv(OUT, index=False)
    print(f"{len(df)} images -> {OUT}")
    print(df.groupby(["split", "label"]).size())


if __name__ == "__main__":
    main()

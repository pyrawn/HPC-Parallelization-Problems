"""
One-time extraction of DIC-C2DH-HeLa.zip for Exercise 2.

The zip lives at the repository root.  Run this script once to extract
all images into exercise_2/data/, then delete it.

Usage:
    python setup_data.py
"""

import zipfile
from pathlib import Path

ZIP_PATH = Path(__file__).parent.parent / "DIC-C2DH-HeLa.zip"
DATA_DIR  = Path(__file__).parent / "data"


def extract() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Zip archive not found: {ZIP_PATH}\n"
            "Place DIC-C2DH-HeLa.zip in the repository root."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {ZIP_PATH.name}  →  {DATA_DIR} …")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)

    raw_tifs = [
        p for p in sorted(DATA_DIR.rglob("*.tif"))
        if "_GT" not in p.parts and "_ST" not in p.parts
           and "_ERR" not in p.parts
    ]
    print(f"\nExtracted {len(raw_tifs)} raw image frames.")
    print("Sample paths:")
    for p in raw_tifs[:6]:
        print(f"  {p.relative_to(DATA_DIR)}")

    print(f"\nAll data in: {DATA_DIR.resolve()}")
    print("You can now delete this script (setup_data.py).")


if __name__ == "__main__":
    extract()

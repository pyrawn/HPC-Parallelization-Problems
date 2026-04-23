"""
One-time download of SuiteSparse matrices needed for Exercise 1.

Matrices
--------
  HB/west0479  – 479 × 479  (chemical-process simulation, 1879 NNZ)
  HB/bcsstk13  – 2003 × 2003 (structural engineering, 83883 NNZ)

Usage
-----
    python download_sparse.py

After the matrices have been downloaded to exercise_1/data/, delete this file.
"""

import tarfile
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data"
BASE_URL  = "https://sparse.tamu.edu/MM"

MATRICES = [
    ("HB", "west0479"),
    ("HB", "bcsstk13"),
]


def download_matrix(group: str, name: str) -> None:
    dest = DATA_DIR / f"{name}.mtx"
    if dest.exists():
        print(f"  {name}.mtx already present – skipping.")
        return

    url = f"{BASE_URL}/{group}/{name}.tar.gz"
    print(f"  Downloading {name} … ", end="", flush=True)
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    tar_path = DATA_DIR / f"{name}.tar.gz"
    with open(tar_path, "wb") as fh:
        for chunk in resp.iter_content(chunk_size=65536):
            fh.write(chunk)
    print("done.")

    print(f"  Extracting {name}.tar.gz … ", end="", flush=True)
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if member.name.endswith(".mtx"):
                member.name = dest.name          # flatten nested directory
                tar.extract(member, DATA_DIR)
                break
    tar_path.unlink()
    print("done.")


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading SuiteSparse matrices into exercise_1/data/\n")
    for group, name in MATRICES:
        download_matrix(group, name)
    print(f"\nAll matrices saved to {DATA_DIR.resolve()}")
    print("You can now delete this file (download_sparse.py).")

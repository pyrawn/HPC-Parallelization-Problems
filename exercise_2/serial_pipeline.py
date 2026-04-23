"""
Tasks 1–5 – Serial cell-image processing pipeline.

Pipeline per image
------------------
  1. Read 16-bit TIFF frame.
  2. Normalise and run Cellpose (cyto2 model) to obtain a label mask.
  3. Extract connected components via regionprops.
  4. Compute: axis-aligned bounding box, area, major/minor axis length
     (in pixels and µm), and a rotated minimum-area bounding box via OpenCV.
  5. Save annotated visualisation for the first few images.

Dataset
-------
  DIC-C2DH-HeLa (Cell Tracking Challenge)
    - 2 sequences (01, 02), 84 frames each  →  168 raw TIFF images
    - Format  : 16-bit grayscale TIFF, 512 × 512 pixels
    - Pixel size: 0.19 µm/pixel  (Cell Tracking Challenge metadata)

Run:
    python serial_pipeline.py [--vis 3]
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops_table

# ── Constants ──────────────────────────────────────────────────────────────────
PIXEL_SIZE_UM = 0.19          # µm per pixel (CTC metadata)
CELLPOSE_DIAMETER = 80        # target cell diameter in pixels for Cellpose
DATA_DIR    = Path(__file__).parent / "data" / "DIC-C2DH-HeLa"
RESULTS_DIR = Path(__file__).parent / "results"

sys.path.insert(0, str(Path(__file__).parent))


# ── Dataset discovery ──────────────────────────────────────────────────────────

def find_images(data_dir: Path = DATA_DIR) -> list[Path]:
    """Return sorted list of raw image paths (sequences 01 and 02 only)."""
    images = []
    for seq in ("01", "02"):
        seq_dir = data_dir / seq
        if seq_dir.exists():
            images.extend(sorted(seq_dir.glob("t*.tif")))
    return images


def inspect_dataset(data_dir: Path = DATA_DIR) -> None:
    """Task 1 – print dataset summary."""
    images = find_images(data_dir)
    if not images:
        print(f"No images found under {data_dir}. Run setup_data.py first.")
        return

    sample = tifffile.imread(str(images[0]))
    print("=" * 55)
    print("  DIC-C2DH-HeLa Dataset Summary")
    print("=" * 55)
    print(f"  Total raw frames : {len(images)}")
    print(f"  Sequences        : 01 ({sum(1 for p in images if '/01/' in str(p).replace(chr(92),'/'))} frames), "
          f"02 ({sum(1 for p in images if '/02/' in str(p).replace(chr(92),'/'))} frames)")
    print(f"  Image dimensions : {sample.shape[1]} × {sample.shape[0]} px")
    print(f"  Bit depth        : {sample.dtype}")
    print(f"  Pixel size       : {PIXEL_SIZE_UM} µm/px (CTC metadata)")
    print(f"  Physical FOV     : {sample.shape[1] * PIXEL_SIZE_UM:.1f} × "
          f"{sample.shape[0] * PIXEL_SIZE_UM:.1f} µm")


# ── Image loading ──────────────────────────────────────────────────────────────

def load_image(path: Path) -> np.ndarray:
    """Load a 16-bit TIFF and return float32 normalised to [0, 1]."""
    img = tifffile.imread(str(path)).astype(np.float32)
    if img.max() > 0:
        img /= img.max()
    return img


# ── Segmentation ───────────────────────────────────────────────────────────────

def load_model():
    """Load the Cellpose cyto2 model (CPU).

    Cellpose 3.x renamed Cellpose → CellposeModel; this wrapper handles both.
    """
    try:
        from cellpose import models
    except ImportError:
        raise ImportError(
            "cellpose is not installed. Run:  pip install cellpose"
        )
    # Cellpose 3.x
    if hasattr(models, "CellposeModel"):
        return models.CellposeModel(gpu=False, model_type="cyto2")
    # Cellpose 2.x fallback
    return models.Cellpose(model_type="cyto2", gpu=False)


def segment(image: np.ndarray, model, diameter: int = CELLPOSE_DIAMETER) -> np.ndarray:
    """
    Task 2 & 3 – Run Cellpose on a single frame.

    Cellpose cyto2 is chosen because it generalises well to DIC microscopy
    without retraining.  channels=[0,0] treats the input as grayscale with
    no separate nucleus channel.

    Returns an integer label mask (0 = background, k = cell k).
    """
    img_u8 = (image * 255).clip(0, 255).astype(np.uint8)
    masks, _, _ = model.eval(img_u8, diameter=diameter, channels=[0, 0])
    return masks.astype(np.int32)


# ── Measurement ────────────────────────────────────────────────────────────────

def measure_cells(mask: np.ndarray, pixel_size: float = PIXEL_SIZE_UM) -> pd.DataFrame:
    """
    Task 4 – Compute per-cell geometric descriptors.

    Columns returned
    ----------------
    label, area (px²), bbox-{0..3}, centroid-{0,1},
    major_axis_length (px), minor_axis_length (px), orientation (rad),
    area_um2, major_axis_um, minor_axis_um,
    bbox_width_px, bbox_height_px, bbox_width_um, bbox_height_um
    """
    if mask.max() == 0:
        return pd.DataFrame()

    props = regionprops_table(
        mask,
        properties=[
            "label", "area",
            "bbox",                  # (min_row, min_col, max_row, max_col)
            "centroid",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
        ],
    )
    df = pd.DataFrame(props)

    df["bbox_width_px"]   = df["bbox-3"] - df["bbox-1"]
    df["bbox_height_px"]  = df["bbox-2"] - df["bbox-0"]
    df["area_um2"]        = df["area"]              * pixel_size ** 2
    df["major_axis_um"]   = df["major_axis_length"] * pixel_size
    df["minor_axis_um"]   = df["minor_axis_length"] * pixel_size
    df["bbox_width_um"]   = df["bbox_width_px"]     * pixel_size
    df["bbox_height_um"]  = df["bbox_height_px"]    * pixel_size
    return df


def rotated_boxes(mask: np.ndarray) -> dict:
    """
    Task 5 – Compute minimum-area rotated bounding rectangles via OpenCV.

    For each cell, the binary mask is converted to a contour with
    cv2.findContours, then cv2.minAreaRect returns (center, (w, h), angle).
    cv2.boxPoints gives the four corner coordinates for drawing.

    Returns a dict mapping cell_id → (center, (w, h), angle_deg).
    """
    boxes = {}
    for cell_id in np.unique(mask):
        if cell_id == 0:
            continue
        cell_bin = (mask == cell_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            cell_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            boxes[int(cell_id)] = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    return boxes


# ── Visualisation ──────────────────────────────────────────────────────────────

def visualize(
    image: np.ndarray,
    mask: np.ndarray,
    df: pd.DataFrame,
    rot_boxes: dict,
    output_path: Path,
) -> None:
    """Save a two-panel annotated figure for one image frame."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    img_u8 = (image * 255).clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # ── Left: axis-aligned bounding boxes ─────────────────────────────────────
    axes[0].imshow(img_u8, cmap="gray")
    axes[0].set_title(f"Axis-aligned bounding boxes  ({len(df)} cells)", fontsize=10)
    for _, row in df.iterrows():
        r0 = int(row["bbox-0"]); c0 = int(row["bbox-1"])
        r1 = int(row["bbox-2"]); c1 = int(row["bbox-3"])
        rect = plt.Rectangle(
            (c0, r0), c1 - c0, r1 - r0,
            edgecolor="cyan", facecolor="none", linewidth=0.8,
        )
        axes[0].add_patch(rect)
    axes[0].axis("off")

    # ── Right: rotated bounding boxes ─────────────────────────────────────────
    axes[1].imshow(img_u8, cmap="gray")
    axes[1].set_title("Rotated min-area bounding boxes  (OpenCV)", fontsize=10)
    for rect in rot_boxes.values():
        pts = cv2.boxPoints(rect).astype(np.float32)   # 4 corners (x, y)
        poly = plt.Polygon(pts, edgecolor="lime", facecolor="none", linewidth=0.8)
        axes[1].add_patch(poly)
    axes[1].axis("off")

    fig.suptitle(f"DIC-C2DH-HeLa  —  {output_path.stem}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ── Full pipeline for one image ────────────────────────────────────────────────

def process_image(
    path: Path,
    model=None,
    pixel_size: float = PIXEL_SIZE_UM,
    diameter: int = CELLPOSE_DIAMETER,
    save_vis: bool = False,
    vis_path: Path | None = None,
) -> dict:
    """
    Load → segment → measure → (optionally visualise) one TIFF frame.

    Returns a dict with keys:
        path (str), n_cells (int), cells (DataFrame), time_s (float)
    """
    t0 = time.perf_counter()

    if model is None:
        model = load_model()

    image    = load_image(path)
    mask     = segment(image, model, diameter=diameter)
    df       = measure_cells(mask, pixel_size=pixel_size)
    rot      = rotated_boxes(mask)

    if save_vis and vis_path is not None:
        visualize(image, mask, df, rot, vis_path)

    return {
        "path":    str(path),
        "n_cells": len(df),
        "cells":   df,
        "time_s":  time.perf_counter() - t0,
    }


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serial cell pipeline")
    parser.add_argument(
        "--vis", type=int, default=3,
        help="Number of images for which to save annotated visualisations (default: 3)"
    )
    args = parser.parse_args()

    inspect_dataset()
    images = find_images()
    if not images:
        sys.exit(1)

    print(f"\nLoading Cellpose model …")
    model = load_model()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []

    print(f"\nProcessing {len(images)} images serially …\n")
    t_total = time.perf_counter()

    for idx, img_path in enumerate(images):
        save_vis = idx < args.vis
        vis_name = f"vis_{img_path.parent.name}_{img_path.stem}.png"
        result = process_image(
            img_path, model=model,
            save_vis=save_vis,
            vis_path=RESULTS_DIR / vis_name if save_vis else None,
        )

        row = {
            "image":      img_path.parent.name + "/" + img_path.name,
            "n_cells":    result["n_cells"],
            "mean_width_um":  result["cells"]["bbox_width_um"].mean()  if result["n_cells"] else float("nan"),
            "std_width_um":   result["cells"]["bbox_width_um"].std()   if result["n_cells"] else float("nan"),
            "mean_length_um": result["cells"]["major_axis_um"].mean()  if result["n_cells"] else float("nan"),
            "std_length_um":  result["cells"]["major_axis_um"].std()   if result["n_cells"] else float("nan"),
            "time_s":     result["time_s"],
        }
        all_rows.append(row)
        print(f"  [{idx+1:3d}/{len(images)}] {row['image']:30s} "
              f"cells={row['n_cells']:3d}  t={row['time_s']:.2f}s")

    total_time = time.perf_counter() - t_total

    summary = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / "serial_summary.csv"
    summary.to_csv(csv_path, index=False)

    print(f"\nTotal serial time : {total_time:.2f} s")
    print(f"Summary saved     : {csv_path}")
    print(f"\nPer-image summary (first 5 rows):")
    print(summary[["image", "n_cells", "mean_length_um", "mean_width_um"]].head())

"""
Task 6 – Parallel cell-image pipeline using Python multiprocessing.

Parallelisation strategy: distribute by image.
-----------------------------------------------
Each image is an independent unit of work (no inter-image data dependency),
so the natural decomposition is one task per image.  A multiprocessing Pool
maps image paths to worker processes.

The Cellpose model is a large neural network that must not be shared across
processes (PyTorch state is not fork-safe on Windows, which uses the 'spawn'
start method).  Instead, the Pool is created with an *initializer* that loads
the model once per worker process, stored in a module-level variable reused
for all images assigned to that worker.  This amortises the ~3–5 s model-load
cost over the worker's lifetime.

Run:
    python parallel_pipeline.py [--workers 4] [--limit 20]
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from serial_pipeline import (
    CELLPOSE_DIAMETER, DATA_DIR, PIXEL_SIZE_UM, RESULTS_DIR,
    find_images, load_model, process_image,
)

# ── Worker state (one model per worker process, loaded by initializer) ─────────

_model = None


def _init_worker() -> None:
    """Load Cellpose once when the worker process starts."""
    global _model
    try:
        import torch
        torch.set_num_threads(1)    # prevent thread contention between workers
    except ImportError:
        pass
    _model = load_model()


def _worker_fn(args: tuple) -> dict:
    """Process one image with the worker's pre-loaded model."""
    path, pixel_size, diameter = args
    return process_image(path, model=_model,
                         pixel_size=pixel_size, diameter=diameter)


# ── Parallel runner ────────────────────────────────────────────────────────────

def run_parallel(
    image_paths: list[Path],
    n_workers: int,
    pixel_size: float = PIXEL_SIZE_UM,
    diameter: int = CELLPOSE_DIAMETER,
) -> tuple[list[dict], float]:
    """
    Process all images in parallel and return (results, wall_clock_time).

    Pool is created fresh for each call so timing includes worker startup
    and model loading (a one-time cost per worker).
    """
    task_args = [(p, pixel_size, diameter) for p in image_paths]
    t0 = time.perf_counter()
    with Pool(processes=n_workers, initializer=_init_worker) as pool:
        results = pool.map(_worker_fn, task_args)
    return results, time.perf_counter() - t0


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel cell pipeline")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 4),
                        help="Number of worker processes")
    parser.add_argument("--limit",   type=int, default=20,
                        help="Maximum number of images to process (default: 20)")
    args = parser.parse_args()

    images = find_images()[:args.limit]
    if not images:
        print("No images found. Run setup_data.py first.")
        sys.exit(1)

    print(f"Parallel pipeline  —  images={len(images)}, workers={args.workers}")

    results, wall_time = run_parallel(images, args.workers)

    n_cells_total = sum(r["n_cells"] for r in results)
    print(f"\nWall-clock time : {wall_time:.2f} s")
    print(f"Images processed: {len(results)}")
    print(f"Total cells     : {n_cells_total}")
    print(f"Avg cells/image : {n_cells_total / max(len(results), 1):.1f}")

    rows = [
        {
            "image":   Path(r["path"]).parent.name + "/" + Path(r["path"]).name,
            "n_cells": r["n_cells"],
            "time_s":  r["time_s"],
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"parallel_w{args.workers}_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary saved   : {csv_path}")

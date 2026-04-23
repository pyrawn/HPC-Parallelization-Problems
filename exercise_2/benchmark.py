"""
Tasks 7 & 8 – Benchmark serial vs parallel cell-image pipeline.

Measures wall-clock time for serial execution and for parallel execution
with different worker counts across a controlled subset of images.
Generates:
    results/timing_comparison.png   – serial vs parallel bar chart
    results/speedup_vs_workers.png  – speedup and efficiency curves
    results/summary_table.csv       – per-image measurements (serial run)
    results/timing_results.csv      – raw timing data

All outputs are also copied to docs/assets/.

Usage (from exercise_2/):
    python benchmark.py
    python benchmark.py --workers 2 4 8 --limit 16 --reps 1
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from serial_pipeline import (
    CELLPOSE_DIAMETER, PIXEL_SIZE_UM, RESULTS_DIR,
    find_images, load_model, process_image,
)
from parallel_pipeline import run_parallel

ASSETS_DIR = Path(__file__).parent.parent / "docs" / "assets"


# ── Serial benchmark ───────────────────────────────────────────────────────────

def run_serial(
    image_paths: list[Path],
    pixel_size: float = PIXEL_SIZE_UM,
    diameter: int = CELLPOSE_DIAMETER,
) -> tuple[list[dict], float]:
    """Load model once, process all images sequentially."""
    model = load_model()
    results = []
    t0 = time.perf_counter()
    for path in image_paths:
        results.append(process_image(path, model=model,
                                     pixel_size=pixel_size, diameter=diameter))
    return results, time.perf_counter() - t0


# ── Summary table (Task 7) ─────────────────────────────────────────────────────

def build_summary(results: list[dict]) -> pd.DataFrame:
    """Build per-image summary: n_cells, mean/std of width and length."""
    rows = []
    for r in results:
        cells = r["cells"]
        if len(cells) > 0:
            rows.append({
                "image":           Path(r["path"]).parent.name + "/" + Path(r["path"]).name,
                "n_cells":         len(cells),
                "mean_width_um":   cells["bbox_width_um"].mean(),
                "std_width_um":    cells["bbox_width_um"].std(),
                "mean_length_um":  cells["major_axis_um"].mean(),
                "std_length_um":   cells["major_axis_um"].std(),
                "mean_area_um2":   cells["area_um2"].mean(),
                "time_s":          r["time_s"],
            })
        else:
            rows.append({
                "image":          Path(r["path"]).parent.name + "/" + Path(r["path"]).name,
                "n_cells": 0, "mean_width_um": float("nan"),
                "std_width_um": float("nan"), "mean_length_um": float("nan"),
                "std_length_um": float("nan"), "mean_area_um2": float("nan"),
                "time_s": r["time_s"],
            })
    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_timing(serial_time: float, parallel_times: dict[int, float]) -> None:
    """Bar chart: serial vs parallel total wall-clock time."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    labels = ["Serial"] + [f"p={w}" for w in sorted(parallel_times)]
    times  = [serial_time] + [parallel_times[w] for w in sorted(parallel_times)]
    colors = ["#4878cf"] + ["#6acc65"] * len(parallel_times)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, times, color=colors, edgecolor="white", width=0.55)
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{t:.1f}s", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("Wall-clock time (s)", fontsize=11)
    ax.set_title("Serial vs Parallel Pipeline — Total Execution Time", fontsize=12)
    ax.set_ylim(0, max(times) * 1.15)
    ax.grid(axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "timing_comparison.png", dpi=150)
    plt.close(fig)
    print("  Saved: timing_comparison.png")


def plot_speedup(serial_time: float, parallel_times: dict[int, float]) -> None:
    """Speedup and efficiency curves vs number of workers."""
    workers = sorted(parallel_times)
    speedups    = [serial_time / parallel_times[w] for w in workers]
    efficiencies = [s / w for s, w in zip(speedups, workers)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.plot(workers, speedups, "o-", color="#2ca02c", linewidth=1.8, label="Measured")
    ax1.plot([workers[0], workers[-1]], [1, workers[-1] / workers[0]],
             "k--", alpha=0.4, label="Ideal")
    ax1.set_xlabel("Workers")
    ax1.set_ylabel("Speedup")
    ax1.set_title("Speedup vs. Workers")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.35)

    ax2.plot(workers, efficiencies, "s-", color="#d62728", linewidth=1.8)
    ax2.axhline(1.0, color="k", linestyle="--", alpha=0.4)
    ax2.set_xlabel("Workers")
    ax2.set_ylabel("Efficiency  (Speedup / Workers)")
    ax2.set_title("Parallel Efficiency vs. Workers")
    ax2.set_ylim(0, 1.15)
    ax2.grid(alpha=0.35)

    fig.suptitle("Cell Pipeline — Scalability", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "speedup_vs_workers.png", dpi=150)
    plt.close(fig)
    print("  Saved: speedup_vs_workers.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise 2 benchmark")
    parser.add_argument("--workers", nargs="+", type=int, default=[2, 4, 8],
                        help="Worker counts for parallel runs (default: 2 4 8)")
    parser.add_argument("--limit",   type=int, default=16,
                        help="Number of images to process (default: 16)")
    parser.add_argument("--reps",    type=int, default=1,
                        help="Repetitions per timing measurement (default: 1)")
    args = parser.parse_args()

    max_cpu = os.cpu_count() or 4
    workers = [w for w in args.workers if w <= max_cpu]
    if not workers:
        workers = [2]

    images = find_images()[:args.limit]
    if not images:
        print("No images found under exercise_2/data/. Run setup_data.py first.")
        sys.exit(1)

    print("=" * 60)
    print("  Exercise 2 — Cell Pipeline Benchmark")
    print("=" * 60)
    print(f"  Images  : {len(images)}")
    print(f"  Workers : {workers}")
    print(f"  Reps    : {args.reps}")

    # ── Serial ─────────────────────────────────────────────────────────────────
    print("\n[Serial]")
    serial_times = []
    for rep in range(args.reps):
        print(f"  Run {rep + 1}/{args.reps} …")
        results_serial, t = run_serial(images)
        serial_times.append(t)
        print(f"    → {t:.2f} s")
    t_serial = min(serial_times)
    print(f"  Best serial time : {t_serial:.2f} s")

    summary = build_summary(results_serial)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
    print(f"\nPer-image summary (first 5 rows):")
    print(summary[["image", "n_cells", "mean_length_um", "mean_width_um"]].head().to_string(index=False))

    # ── Parallel ───────────────────────────────────────────────────────────────
    parallel_times: dict[int, float] = {}
    timing_rows = [{"method": "serial", "workers": 1, "time_s": t_serial}]

    for w in workers:
        print(f"\n[Parallel  workers={w}]")
        run_times = []
        for rep in range(args.reps):
            print(f"  Run {rep + 1}/{args.reps} …")
            _, t = run_parallel(images, w)
            run_times.append(t)
            print(f"    → {t:.2f} s")
        t_best = min(run_times)
        parallel_times[w] = t_best
        speedup = t_serial / t_best
        efficiency = speedup / w
        print(f"  Best parallel time : {t_best:.2f} s  "
              f"(speedup {speedup:.2f}×, efficiency {efficiency:.2%})")
        timing_rows.append({"method": "parallel", "workers": w, "time_s": t_best})

    # ── Save timing CSV ────────────────────────────────────────────────────────
    timing_df = pd.DataFrame(timing_rows)
    timing_df["speedup"]    = t_serial / timing_df["time_s"]
    timing_df["efficiency"] = timing_df["speedup"] / timing_df["workers"]
    timing_df.to_csv(RESULTS_DIR / "timing_results.csv", index=False)

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_timing(t_serial, parallel_times)
    if len(workers) > 1:
        plot_speedup(t_serial, parallel_times)

    # ── Copy to docs/assets/ ───────────────────────────────────────────────────
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    for src in list(RESULTS_DIR.glob("*.png")) + list(RESULTS_DIR.glob("*.csv")):
        shutil.copy2(src, ASSETS_DIR / src.name)
    print(f"\n  Assets copied to {ASSETS_DIR.resolve()}")
    print("\nDone.")


if __name__ == "__main__":
    main()

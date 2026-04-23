"""
Tasks 7 & 9 – Comprehensive benchmark for Exercise 1.

Runs all non-MPI methods across matrix sizes and worker counts, then generates
three PNG plots and a CSV summary in exercise_1/results/.

MPI results are loaded automatically if results/mpi_results.json exists
(produced by running mpi_matmul.py separately).

Usage
-----
    # From the exercise_1 directory:
    python benchmark.py
    python benchmark.py --sizes 128 256 512 --workers 2 4 --reps 5
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── make imports work from any working directory ───────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from serial_matmul   import matmul_numpy
from parallel_matmul import (
    matmul_block_parallel,
    matmul_col_parallel,
    matmul_row_parallel,
)
from strassen_matmul import matmul_strassen

SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"
ASSETS_DIR  = Path(__file__).parent.parent / "docs" / "assets"


# ── timing ─────────────────────────────────────────────────────────────────────

def _time(fn, *args, reps: int = 3) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best = elapsed
    return best


# ── benchmark loop ─────────────────────────────────────────────────────────────

def run_benchmarks(sizes: list[int], workers: list[int], reps: int) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    records: list[dict] = []

    for n in sizes:
        print(f"\n  n = {n}")
        A = rng.random((n, n))
        B = rng.random((n, n))

        # serial numpy
        t_np = _time(matmul_numpy, A, B, reps=reps)
        records.append({"method": "numpy", "n": n, "workers": 1, "time": t_np})
        print(f"    numpy          : {t_np:.4f}s")

        # Strassen
        t_st = _time(matmul_strassen, A, B, reps=reps)
        records.append({"method": "strassen", "n": n, "workers": 1, "time": t_st})
        print(f"    strassen       : {t_st:.4f}s")

        for w in workers:
            # row
            t = _time(matmul_row_parallel, A, B, w, reps=reps)
            speedup = t_np / t
            records.append({"method": "row", "n": n, "workers": w, "time": t})
            print(f"    row   w={w:<2}     : {t:.4f}s  (speedup {speedup:.2f}×)")

            # col
            t = _time(matmul_col_parallel, A, B, w, reps=reps)
            speedup = t_np / t
            records.append({"method": "col", "n": n, "workers": w, "time": t})
            print(f"    col   w={w:<2}     : {t:.4f}s  (speedup {speedup:.2f}×)")

            # block
            t = _time(matmul_block_parallel, A, B, w, reps=reps)
            speedup = t_np / t
            records.append({"method": "block", "n": n, "workers": w, "time": t})
            print(f"    block w={w:<2}     : {t:.4f}s  (speedup {speedup:.2f}×)")

    return pd.DataFrame(records)


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_all(df: pd.DataFrame, mpi_path: Path | None) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sizes   = sorted(df["n"].unique())
    workers = sorted(df[df["method"] == "row"]["workers"].unique())

    _COLORS = {
        "numpy":    "#1f77b4",
        "strassen": "#ff7f0e",
        "row":      "#2ca02c",
        "col":      "#d62728",
        "block":    "#9467bd",
        "mpi":      "#8c564b",
    }

    # ── 1. Execution time vs matrix size (best worker count per parallel method) ──
    fig, ax = plt.subplots(figsize=(9, 5))

    t_np = df[df["method"] == "numpy"].set_index("n")["time"].reindex(sizes)
    ax.plot(sizes, t_np, "o-", color=_COLORS["numpy"], label="NumPy (serial)")

    t_st = df[df["method"] == "strassen"].set_index("n")["time"].reindex(sizes)
    ax.plot(sizes, t_st, "s--", color=_COLORS["strassen"], label="Strassen")

    for method in ("row", "col", "block"):
        best = (
            df[df["method"] == method]
            .groupby("n")["time"]
            .min()
            .reindex(sizes)
        )
        ax.plot(sizes, best, "^-", color=_COLORS[method],
                label=f"{method.capitalize()} (best workers)")

    if mpi_path and mpi_path.exists():
        mpi_df = pd.DataFrame(json.loads(mpi_path.read_text()))
        for nproc in sorted(mpi_df["nprocs"].unique()):
            sub = mpi_df[mpi_df["nprocs"] == nproc].sort_values("size")
            ax.plot(sub["size"], sub["min_time"], "d:",
                    color=_COLORS["mpi"], label=f"MPI (p={nproc})")

    ax.set_xlabel("Matrix size n (n×n matrix)", fontsize=11)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.set_title("Execution Time vs. Matrix Size", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = RESULTS_DIR / "time_vs_size.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out}")

    # ── 2. Speedup vs number of workers ───────────────────────────────────────
    n_panels = len(sizes)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=False)
    if n_panels == 1:
        axes = [axes]

    for ax, n in zip(axes, sizes):
        t_serial = float(df[(df["method"] == "numpy") & (df["n"] == n)]["time"].iloc[0])
        ax.plot([workers[0], workers[-1]], [1, workers[-1] / workers[0]],
                "k--", alpha=0.35, label="Ideal")

        for method, marker in [("row", "o"), ("col", "s"), ("block", "^")]:
            sub = (
                df[(df["method"] == method) & (df["n"] == n)]
                .sort_values("workers")
            )
            speedups = t_serial / sub["time"].values
            ax.plot(sub["workers"], speedups, f"{marker}-",
                    color=_COLORS[method], label=method.capitalize())

        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_xlabel("Workers", fontsize=9)
        ax.set_ylabel("Speedup", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Speedup vs. Number of Workers", fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / "speedup_vs_workers.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # ── 3. Efficiency vs number of workers ────────────────────────────────────
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, n in zip(axes, sizes):
        t_serial = float(df[(df["method"] == "numpy") & (df["n"] == n)]["time"].iloc[0])
        ax.axhline(1.0, color="k", linestyle="--", alpha=0.35, label="Ideal")

        for method, marker in [("row", "o"), ("col", "s"), ("block", "^")]:
            sub = (
                df[(df["method"] == method) & (df["n"] == n)]
                .sort_values("workers")
            )
            eff = (t_serial / sub["time"].values) / sub["workers"].values
            ax.plot(sub["workers"], eff, f"{marker}-",
                    color=_COLORS[method], label=method.capitalize())

        ax.set_title(f"n = {n}", fontsize=10)
        ax.set_xlabel("Workers", fontsize=9)
        ax.set_ylabel("Efficiency", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Parallel Efficiency vs. Number of Workers", fontsize=13)
    fig.tight_layout()
    out = RESULTS_DIR / "efficiency_vs_workers.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise 1 master benchmark"
    )
    parser.add_argument(
        "--sizes",   nargs="+", type=int, default=[128, 256, 512, 1024],
        help="Matrix sizes to benchmark (default: 128 256 512 1024)"
    )
    parser.add_argument(
        "--workers", nargs="+", type=int, default=[2, 4, 8],
        help="Worker counts to test (default: 2 4 8)"
    )
    parser.add_argument(
        "--reps",    type=int,  default=3,
        help="Number of repetitions per measurement (default: 3)"
    )
    args = parser.parse_args()

    max_cpu = os.cpu_count() or 4
    workers = [w for w in args.workers if w <= max_cpu]
    if not workers:
        workers = [2]

    print("=" * 60)
    print("  Exercise 1 – Matrix Multiplication Benchmark")
    print("=" * 60)
    print(f"  sizes   = {args.sizes}")
    print(f"  workers = {workers}")
    print(f"  reps    = {args.reps}")
    print(f"  CPUs    = {max_cpu}")

    df = run_benchmarks(args.sizes, workers, args.reps)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = RESULTS_DIR / "benchmark_results.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n  CSV saved: {csv_out}")

    mpi_json = RESULTS_DIR / "mpi_results.json"
    print("\nGenerating plots …")
    plot_all(df, mpi_json if mpi_json.exists() else None)

    # ── Copy outputs to docs/assets/ for the report ───────────────────────────
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = list(RESULTS_DIR.glob("*.png")) + list(RESULTS_DIR.glob("*.csv"))
    if mpi_json.exists():
        artifacts.append(mpi_json)
    for src in artifacts:
        shutil.copy2(src, ASSETS_DIR / src.name)
    print(f"\n  Assets copied to {ASSETS_DIR.resolve()}")
    print("\nDone.")


if __name__ == "__main__":
    main()

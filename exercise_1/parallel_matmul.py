"""
Tasks 2–4 – Parallel matrix multiplication with Python multiprocessing.

Three decomposition strategies:
  matmul_row_parallel   – strips of rows of A distributed across workers
  matmul_col_parallel   – strips of columns of B distributed across workers
  matmul_block_parallel – 2-D grid of (row-strip × col-strip) blocks

All worker functions are defined at module level so they are picklable on
every platform, including Windows (which uses the 'spawn' start method).

Run standalone:
    python parallel_matmul.py
"""

import math
import os
import time
from multiprocessing import Pool

import numpy as np

SEED = 42


# ── Row partition ──────────────────────────────────────────────────────────────

def _row_worker(args: tuple) -> np.ndarray:
    A_chunk, B = args
    return A_chunk @ B


def matmul_row_parallel(A: np.ndarray, B: np.ndarray, n_workers: int) -> np.ndarray:
    """
    C = A @ B with row decomposition.

    A is split horizontally into n_workers strips; each worker computes its
    strip of rows of C independently. Results are stacked vertically.
    """
    chunks = np.array_split(A, n_workers, axis=0)
    with Pool(n_workers) as pool:
        parts = pool.map(_row_worker, [(c, B) for c in chunks])
    return np.vstack(parts)


# ── Column partition ───────────────────────────────────────────────────────────

def _col_worker(args: tuple) -> np.ndarray:
    A, B_chunk = args
    return A @ B_chunk


def matmul_col_parallel(A: np.ndarray, B: np.ndarray, n_workers: int) -> np.ndarray:
    """
    C = A @ B with column decomposition.

    B is split vertically into n_workers strips; each worker computes its
    strip of columns of C independently. Results are stacked horizontally.
    """
    chunks = np.array_split(B, n_workers, axis=1)
    with Pool(n_workers) as pool:
        parts = pool.map(_col_worker, [(A, c) for c in chunks])
    return np.hstack(parts)


# ── Block (2-D) partition ──────────────────────────────────────────────────────

def _block_worker(args: tuple) -> tuple:
    i, j, A_rows, B_cols = args
    return i, j, A_rows @ B_cols


def matmul_block_parallel(A: np.ndarray, B: np.ndarray, n_workers: int) -> np.ndarray:
    """
    C = A @ B with a 2-D block decomposition.

    The grid size g = floor(sqrt(n_workers)) is chosen so that worker (i, j)
    receives only A[row_i, :] and B[:, col_j], reducing the data sent to each
    worker compared with the row partition (which broadcasts full B).

    Block (i, j) computes C[row_i, col_j] = A[row_i, :] @ B[:, col_j] with
    no accumulation step required.
    """
    g = max(1, int(math.isqrt(n_workers)))
    row_chunks = np.array_split(A, g, axis=0)
    col_chunks = np.array_split(B, g, axis=1)

    tasks = [
        (i, j, row_chunks[i], col_chunks[j])
        for i in range(g)
        for j in range(g)
    ]

    actual_workers = min(len(tasks), n_workers)
    with Pool(actual_workers) as pool:
        results = pool.map(_block_worker, tasks)

    grid: dict = {}
    for i, j, block in results:
        grid[(i, j)] = block

    return np.vstack([
        np.hstack([grid[(i, j)] for j in range(g)])
        for i in range(g)
    ])


# ── Timing helper ──────────────────────────────────────────────────────────────

def _time(fn, *args, reps: int = 3) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best = elapsed
    return best


if __name__ == "__main__":
    np.random.seed(SEED)
    n_workers = min(4, os.cpu_count() or 4)

    print("=" * 55)
    print(f"  Parallel Matrix Multiplication  (workers={n_workers})")
    print("=" * 55)

    # ── Correctness ────────────────────────────────────────────────────────────
    print("\nCorrectness validation:")
    for n in [64, 128, 256]:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        ref = A @ B
        for name, fn in [
            ("row",   lambda a, b: matmul_row_parallel(a, b, n_workers)),
            ("col",   lambda a, b: matmul_col_parallel(a, b, n_workers)),
            ("block", lambda a, b: matmul_block_parallel(a, b, n_workers)),
        ]:
            ok = np.allclose(fn(A, B), ref, atol=1e-8)
            print(f"  n={n:4d} | {name:5s}: {'PASS' if ok else 'FAIL'}")

    # ── Performance ────────────────────────────────────────────────────────────
    print(f"\nPerformance vs NumPy (best of 3, seconds)  [workers={n_workers}]:")
    hdr = f"{'Size':>6} | {'NumPy':>8} | {'Row':>8} | {'Col':>8} | {'Block':>8}"
    print(hdr)
    print("-" * len(hdr))

    for n in [256, 512, 1024]:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        t_np    = _time(lambda a, b: a @ b, A, B)
        t_row   = _time(matmul_row_parallel,   A, B, n_workers)
        t_col   = _time(matmul_col_parallel,   A, B, n_workers)
        t_block = _time(matmul_block_parallel, A, B, n_workers)
        print(
            f"{n:>6} | {t_np:>8.4f} | {t_row:>8.4f} | {t_col:>8.4f} | {t_block:>8.4f}"
        )

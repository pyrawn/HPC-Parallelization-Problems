"""
Task 1 – Serial matrix multiplication baseline.

Two implementations:
  matmul_naive  – pure-Python triple loop (correctness reference, small sizes only)
  matmul_numpy  – NumPy @ operator (optimised serial baseline for benchmarks)

Run standalone:
    python serial_matmul.py
"""

import time

import numpy as np

SEED = 42


def matmul_naive(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Triple-loop serial multiplication – O(n^3) pure Python."""
    m, n = A.shape
    _, p = B.shape
    C = np.zeros((m, p), dtype=np.float64)
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C


def matmul_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """NumPy optimised matrix multiplication (BLAS-backed serial reference)."""
    return A @ B


def validate(A: np.ndarray, B: np.ndarray, C: np.ndarray, tol: float = 1e-8) -> bool:
    """Return True when C is element-wise close to np.dot(A, B)."""
    return bool(np.allclose(C, np.dot(A, B), atol=tol))


def time_fn(fn, *args, reps: int = 3) -> tuple:
    """Return (best_time, last_result) over *reps* runs."""
    best, result = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        result = fn(*args)
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best = elapsed
    return best, result


if __name__ == "__main__":
    np.random.seed(SEED)
    print("=" * 50)
    print("  Serial Matrix Multiplication Baseline")
    print("=" * 50)

    # ── Correctness ────────────────────────────────────────────────────────────
    print("\nCorrectness (naive vs numpy.dot):")
    for n in [4, 8, 16, 32]:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        C = matmul_naive(A, B)
        status = "PASS" if validate(A, B, C) else "FAIL"
        print(f"  n={n:3d}: {status}")

    # ── Performance ────────────────────────────────────────────────────────────
    print("\nPerformance (best of 3 runs, seconds):")
    header = f"{'Size':>6} | {'NumPy':>12} | {'Naive':>12}"
    print(header)
    print("-" * len(header))

    for n in [32, 64, 128, 256, 512, 1024]:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        t_np, _ = time_fn(matmul_numpy, A, B)
        if n <= 64:
            t_naive, _ = time_fn(matmul_naive, A, B, reps=1)
            print(f"{n:>6} | {t_np:>12.6f} | {t_naive:>12.6f}")
        else:
            print(f"{n:>6} | {t_np:>12.6f} | {'(skipped)':>12}")

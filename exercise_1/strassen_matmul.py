"""
Task 6 – Strassen matrix multiplication.

The recursive algorithm reduces the naive O(n^3) complexity to O(n^{2.807})
by replacing one of the eight sub-multiplications with additional additions.

Implementation notes
--------------------
* The algorithm only applies directly to square matrices whose dimension is a
  power of two. Arbitrary inputs are zero-padded to the next shared power-of-2
  dimension and the excess rows/columns are trimmed from the result.
* A *threshold* controls the recursion cutoff: below that size, NumPy's BLAS
  multiply is faster than Strassen's overhead, so we fall back to A @ B.

Run standalone:
    python strassen_matmul.py
"""

import time

import numpy as np

SEED = 42
DEFAULT_THRESHOLD = 64


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _strassen_sq(A: np.ndarray, B: np.ndarray, threshold: int) -> np.ndarray:
    """Strassen on square power-of-2 matrices (internal recursive call)."""
    n = A.shape[0]
    if n <= threshold:
        return A @ B

    h = n // 2
    A11, A12 = A[:h, :h], A[:h, h:]
    A21, A22 = A[h:, :h], A[h:, h:]
    B11, B12 = B[:h, :h], B[:h, h:]
    B21, B22 = B[h:, :h], B[h:, h:]

    M1 = _strassen_sq(A11 + A22, B11 + B22, threshold)
    M2 = _strassen_sq(A21 + A22, B11,       threshold)
    M3 = _strassen_sq(A11,       B12 - B22, threshold)
    M4 = _strassen_sq(A22,       B21 - B11, threshold)
    M5 = _strassen_sq(A11 + A12, B22,       threshold)
    M6 = _strassen_sq(A21 - A11, B11 + B12, threshold)
    M7 = _strassen_sq(A12 - A22, B21 + B22, threshold)

    C = np.empty_like(A)
    C[:h, :h] = M1 + M4 - M5 + M7
    C[:h, h:] = M3 + M5
    C[h:, :h] = M2 + M4
    C[h:, h:] = M1 - M2 + M3 + M6
    return C


def matmul_strassen(
    A: np.ndarray, B: np.ndarray, threshold: int = DEFAULT_THRESHOLD
) -> np.ndarray:
    """
    Strassen's algorithm for arbitrary rectangular matrices.

    Steps:
      1. Find the next power-of-2 that covers all three dimensions (m, n, p).
      2. Zero-pad both operands to that dimension.
      3. Recurse with _strassen_sq.
      4. Trim the result back to (m, p).
    """
    m, n = A.shape
    n2, p = B.shape
    if n != n2:
        raise ValueError(f"Incompatible shapes: ({m},{n}) @ ({n2},{p})")

    dim = _next_pow2(max(m, n, p))
    Ap = np.zeros((dim, dim), dtype=np.float64)
    Bp = np.zeros((dim, dim), dtype=np.float64)
    Ap[:m, :n] = A
    Bp[:n, :p] = B

    Cp = _strassen_sq(Ap, Bp, threshold)
    return Cp[:m, :p]


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
    print("=" * 55)
    print("  Strassen Matrix Multiplication")
    print("=" * 55)

    # ── Correctness ────────────────────────────────────────────────────────────
    print("\nCorrectness (vs numpy.dot, atol=1e-8):")
    for n in [4, 8, 16, 32, 64, 100, 200, 300]:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        ok = np.allclose(matmul_strassen(A, B), A @ B, atol=1e-8)
        print(f"  n={n:3d}: {'PASS' if ok else 'FAIL'}")

    # ── Performance vs threshold ───────────────────────────────────────────────
    print("\nExecution time (seconds, best of 3):")
    hdr = f"{'Size':>6} | {'NumPy':>8} | {'Strassen(32)':>13} | {'Strassen(64)':>13} | {'Strassen(128)':>14}"
    print(hdr)
    print("-" * len(hdr))

    for n in [128, 256, 512, 1024]:
        A = np.random.rand(n, n)
        B = np.random.rand(n, n)
        t_np  = _time(lambda a, b: a @ b, A, B)
        t_s32  = _time(matmul_strassen, A, B, 32)
        t_s64  = _time(matmul_strassen, A, B, 64)
        t_s128 = _time(matmul_strassen, A, B, 128)
        print(f"{n:>6} | {t_np:>8.4f} | {t_s32:>13.4f} | {t_s64:>13.4f} | {t_s128:>14.4f}")

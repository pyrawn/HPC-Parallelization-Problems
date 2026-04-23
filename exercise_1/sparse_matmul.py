"""
Task 8 – Sparse matrix analysis using SuiteSparse matrices.

Matrices used
-------------
  west0479  : 479 × 479, chemical-process simulation (HB group)
  bcsstk13  : 2003 × 2003, structural engineering (HB group)

For each matrix we compare:
  (a) dense A_dense @ B_dense          (NumPy)
  (b) sparse A_csr  @ B_dense          (SciPy CSR × dense)
  (c) sparse A_csr  @ A_csr.T          (SciPy SpSpMM)

and report memory footprint, non-zero density, and timing.

Run:
    python sparse_matmul.py
"""

import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread

SEED = 42
DATA_DIR = Path(__file__).parent / "data"

MATRICES = {
    "west0479": "west0479",
    "bcsstk13": "bcsstk13",
}


def load_sparse(name: str) -> sp.csr_matrix:
    path = DATA_DIR / f"{name}.mtx"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Run  python download_sparse.py  first."
        )
    return sp.csr_matrix(mmread(str(path)), dtype=np.float64)


def print_info(M: sp.csr_matrix, label: str) -> None:
    nrows, ncols = M.shape
    nnz   = M.nnz
    total = nrows * ncols
    density = nnz / total
    mem_csr   = (M.data.nbytes + M.indices.nbytes + M.indptr.nbytes) / 1024
    mem_dense = M.toarray().nbytes / 1024
    print(f"\n  {label}")
    print(f"    Shape   : {nrows} × {ncols}")
    print(f"    NNZ     : {nnz:,}")
    print(f"    Density : {density:.4%}")
    print(f"    Memory  : {mem_csr:.1f} KB (CSR)  vs  {mem_dense:.1f} KB (dense)")


def _time(fn, reps: int = 5) -> float:
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best = elapsed
    return best


def benchmark(M_csr: sp.csr_matrix, label: str) -> dict:
    nrows, ncols = M_csr.shape
    density = M_csr.nnz / (nrows * ncols)
    M_dense = M_csr.toarray()
    rng = np.random.default_rng(SEED)
    B   = rng.random((ncols, min(ncols, 256)))

    t_dd   = _time(lambda: M_dense @ B)
    t_sd   = _time(lambda: M_csr   @ B)
    t_spsp = _time(lambda: M_csr   @ M_csr.T)

    speedup = t_dd / t_sd if t_sd > 0 else float("nan")
    print(f"\n  Benchmark — {label}  (density={density:.4%})")
    print(f"    dense  @ dense_B : {t_dd:.6f} s")
    print(f"    sparse @ dense_B : {t_sd:.6f} s   (speedup vs dense: {speedup:.2f}×)")
    print(f"    sparse @ sparse.T: {t_spsp:.6f} s")

    return {
        "name":    label,
        "shape":   list(M_csr.shape),
        "nnz":     M_csr.nnz,
        "density": density,
        "t_dense_x_dense": t_dd,
        "t_sparse_x_dense": t_sd,
        "t_sparse_x_sparse": t_spsp,
        "speedup_sparse_over_dense": speedup,
    }


if __name__ == "__main__":
    np.random.seed(SEED)
    print("=" * 55)
    print("  Sparse Matrix Analysis")
    print("=" * 55)

    records = []
    for name, fname in MATRICES.items():
        try:
            M = load_sparse(fname)
            print_info(M, name)
            records.append(benchmark(M, name))
        except FileNotFoundError as exc:
            print(f"\n  [SKIP] {exc}")

    if records:
        print("\n" + "=" * 55)
        print("  Summary")
        print("=" * 55)
        hdr = f"{'Matrix':>12} | {'Density':>10} | {'Dense(s)':>10} | {'Sparse(s)':>10} | {'Speedup':>8}"
        print(hdr)
        print("-" * len(hdr))
        for r in records:
            print(
                f"{r['name']:>12} | {r['density']:>10.4%} | "
                f"{r['t_dense_x_dense']:>10.6f} | {r['t_sparse_x_dense']:>10.6f} | "
                f"{r['speedup_sparse_over_dense']:>8.2f}×"
            )

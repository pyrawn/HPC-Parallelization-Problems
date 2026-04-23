"""
Task 5 – Distributed matrix multiplication with mpi4py.

Communication pattern
---------------------
  1. Root (rank 0) generates A and B.
  2. B is broadcast to every rank (each rank needs the full B to compute its rows).
  3. A is scattered in row strips using comm.scatter (Python-level, handles
     unequal chunk sizes automatically).
  4. Each rank computes local_C = local_A @ B.
  5. Partial results are gathered at root with comm.gather and stacked.

Run:
    mpiexec -n 4 python mpi_matmul.py
    mpiexec -n 4 python mpi_matmul.py --sizes 256 512 1024 --reps 3 --output results/mpi_results.json

Windows note: requires Microsoft MPI (ms-mpi) installed separately.
    Download: https://www.microsoft.com/en-us/download/details.aspx?id=57467
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

SEED = 42


def mpi_matmul(size: int, comm: MPI.Comm) -> np.ndarray | None:
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if rank == 0:
        rng = np.random.default_rng(SEED)
        A = rng.random((size, size))
        B = rng.random((size, size))
        row_chunks = np.array_split(A, nprocs, axis=0)
    else:
        B = np.empty((size, size), dtype=np.float64)
        row_chunks = None

    # Step 1 – broadcast B
    comm.Bcast(B, root=0)

    # Step 2 – scatter row strips of A
    local_A = comm.scatter(row_chunks, root=0)

    # Step 3 – local multiply
    local_C = local_A @ B

    # Step 4 – gather
    all_C = comm.gather(local_C, root=0)

    if rank == 0:
        return np.vstack(all_C)
    return None


def benchmark(size: int, comm: MPI.Comm, reps: int) -> dict:
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if rank == 0:
        rng = np.random.default_rng(SEED)
        A = rng.random((size, size))
        B = rng.random((size, size))
        row_chunks = np.array_split(A, nprocs, axis=0)
    else:
        B = np.empty((size, size), dtype=np.float64)
        row_chunks = None

    times = []
    for _ in range(reps):
        comm.Barrier()
        t0 = MPI.Wtime()

        comm.Bcast(B, root=0)
        local_A = comm.scatter(row_chunks, root=0)
        local_C = local_A @ B
        all_C = comm.gather(local_C, root=0)

        comm.Barrier()
        times.append(MPI.Wtime() - t0)

    return {
        "size": size,
        "nprocs": nprocs,
        "min_time": float(min(times)),
        "mean_time": float(np.mean(times)),
    }


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    parser = argparse.ArgumentParser(
        description="MPI matrix multiplication benchmark"
    )
    parser.add_argument("--sizes",  nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--reps",   type=int,  default=3)
    parser.add_argument("--output", type=str,  default="results/mpi_results.json")
    args = parser.parse_args()

    if rank == 0:
        print(f"MPI matrix multiplication  [nprocs={comm.Get_size()}]")
        print(f"{'Size':>6} | {'Min time (s)':>14} | {'Mean time (s)':>14}")
        print("-" * 42)

    results = []
    for size in args.sizes:
        rec = benchmark(size, comm, args.reps)
        results.append(rec)
        if rank == 0:
            print(f"{size:>6} | {rec['min_time']:>14.4f} | {rec['mean_time']:>14.4f}")

    if rank == 0:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if out.exists():
            with open(out) as fh:
                existing = json.load(fh)
        existing.extend(results)
        with open(out, "w") as fh:
            json.dump(existing, fh, indent=2)
        print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()

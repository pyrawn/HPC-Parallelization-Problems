# HPC Parallelization Problems — Unit 3 Final Assignment

## Objective

This repository contains four parallel computing exercises that demonstrate how
different parallelisation strategies (Python `multiprocessing` and `mpi4py`)
affect performance for representative scientific-computing workloads. For each
exercise a serial baseline is provided alongside one or more parallel
implementations, and reproducible experiments are used to measure speedup and
efficiency.

---

## Repository Structure

```text
HPC-Parallelization-Problems/
├── requirements.txt
├── README.md
├── exercise_1/               # Parallel Matrix Multiplication
│   ├── serial_matmul.py      # Task 1 – naive triple-loop + NumPy baseline
│   ├── parallel_matmul.py    # Tasks 2-4 – row / col / block (multiprocessing)
│   ├── mpi_matmul.py         # Task 5 – scatter-broadcast-gather (mpi4py)
│   ├── strassen_matmul.py    # Task 6 – recursive Strassen algorithm
│   ├── sparse_matmul.py      # Task 8 – SuiteSparse matrix analysis
│   ├── benchmark.py          # Tasks 7 & 9 – full benchmark + plots
│   ├── data/                 # Sparse .mtx files (populated by download step)
│   └── results/              # Generated plots and CSV (populated by benchmark)
├── exercise_2/               # Cell Image Processing (coming next)
├── exercise_3/               # Forest Fire Cellular Automaton (coming next)
├── exercise_4/               # Parallel K-Means Clustering (coming next)
└── docs/
    ├── assets/               # Figures, tables, and logs for the report (auto-populated)
    └── report.tex            # LaTeX source for the PDF report
```

---

## Software Requirements

| Package | Purpose |
| --- | --- |
| `numpy` | Array operations and serial reference |
| `scipy` | Sparse matrix I/O and SpMM |
| `matplotlib` | Plot generation |
| `pandas` | Benchmark result tables |
| `mpi4py` | Distributed-memory MPI |
| `requests` | One-time sparse matrix download |
| `tqdm` | Progress bars |
| `tabulate` | Pretty console tables |

**MPI runtime (Windows):** `mpi4py` requires Microsoft MPI to be installed
separately.  
Download: <https://www.microsoft.com/en-us/download/details.aspx?id=57467>

**LaTeX (report):** any standard TeX distribution (TeX Live, MiKTeX) with
`pdflatex`.

---

## Environment Setup

```bash
# 1. Create and activate the virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install all dependencies
pip install -r requirements.txt
```

---

## Exercise 1 — Parallel Matrix Multiplication

### 1. Download sparse matrices (run once, then delete the script)

```bash
cd exercise_1
python download_sparse.py
# → saves west0479.mtx and bcsstk13.mtx to exercise_1/data/
# → delete download_sparse.py afterwards
```

### 2. Serial baseline

```bash
python serial_matmul.py
```

Validates correctness with small matrices and prints timing for sizes up
to 1024 × 1024.

### 3. Parallel variants (multiprocessing)

```bash
python parallel_matmul.py
```

Runs row, column, and block decompositions and prints timing for sizes
256 – 1024 using the default worker count (`min(4, cpu_count)`).

### 4. MPI distributed version

```bash
# Run with 4 MPI processes, sizes 256 512 1024
mpiexec -n 4 python mpi_matmul.py --sizes 256 512 1024 --reps 3 --output results/mpi_results.json

# Repeat for other process counts to populate the comparison plot
mpiexec -n 2 python mpi_matmul.py --sizes 256 512 1024 --reps 3 --output results/mpi_results.json
mpiexec -n 8 python mpi_matmul.py --sizes 256 512 1024 --reps 3 --output results/mpi_results.json
```

### 5. Strassen algorithm

```bash
python strassen_matmul.py
```

### 6. Sparse matrix analysis

```bash
python sparse_matmul.py
```

### 7. Full benchmark (generates all plots and CSV)

```bash
# Default: sizes 128 256 512 1024, workers 2 4 8, reps 3
python benchmark.py

# Custom example
python benchmark.py --sizes 256 512 1024 --workers 2 4 8 --reps 5
```

Outputs:

- `results/benchmark_results.csv`
- `results/time_vs_size.png`
- `results/speedup_vs_workers.png`
- `results/efficiency_vs_workers.png`

**Recommended order for a full reproduction:**

```bash
cd exercise_1
python download_sparse.py          # once
python serial_matmul.py
python parallel_matmul.py
python strassen_matmul.py
python sparse_matmul.py
mpiexec -n 2 python mpi_matmul.py --sizes 256 512 1024 --output results/mpi_results.json
mpiexec -n 4 python mpi_matmul.py --sizes 256 512 1024 --output results/mpi_results.json
mpiexec -n 8 python mpi_matmul.py --sizes 256 512 1024 --output results/mpi_results.json
python benchmark.py
```

---

## Exercise 2 — Parallel Cell Image Processing (DIC-C2DH-HeLa)

### 1. Extract the dataset (run once, then delete the script)

```bash
cd exercise_2
python setup_data.py
# → extracts DIC-C2DH-HeLa.zip into exercise_2/data/
# → delete setup_data.py afterwards
```

### 2. Serial pipeline (segmentation + measurement)

```bash
python serial_pipeline.py          # processes all 168 images, saves serial_summary.csv
python serial_pipeline.py --vis 5  # also save annotated visualisations for first 5 images
```

### 3. Parallel pipeline (standalone demo)

```bash
python parallel_pipeline.py --workers 4 --limit 20
```

### 4. Full benchmark (timing comparison + plots)

```bash
# Default: 16 images, workers 2 4 8
python benchmark.py

# Custom example
python benchmark.py --workers 2 4 8 --limit 20 --reps 1
```

Outputs (also copied to `docs/assets/`):

- `results/summary_table.csv` — per-image cell counts and size statistics
- `results/timing_results.csv` — serial vs parallel timing data
- `results/timing_comparison.png` — bar chart
- `results/speedup_vs_workers.png` — speedup and efficiency curves

**Recommended run order:**

```bash
cd exercise_2
python setup_data.py               # once — then delete
python serial_pipeline.py --vis 3  # serial run + visualisations
python benchmark.py                # timing benchmark
```

### Results

Cellpose `cyto2` detected **10–23 HeLa cells per frame** across 168 frames (two sequences of 84). Mean bounding-box width ranged from ~24 to ~30 µm, consistent with the expected HeLa cell diameter at 0.19 µm/px resolution.

| Configuration | Time (s) | Speedup | Efficiency |
|---------------|----------|---------|------------|
| Serial (1)    | 75.25    | 1.000   | 1.000      |
| Parallel 2    | 95.96    | 0.784   | 0.392      |
| Parallel 4    | 79.02    | 0.952   | 0.238      |
| Parallel 6    | 95.92    | 0.785   | 0.131      |

> **Note:** Parallel configurations did not outperform serial for small batches (~16 images). On Windows, spawning each worker process requires reloading the Cellpose model (~200 MB), a fixed startup cost that dominates at this batch size. Speedup is expected to improve for larger batches (>100 frames).

---

## Exercise 3 — Forest Fire Cellular Automaton

### Context

A 2D cellular automaton on an **800 × 800** grid models forest-fire propagation.
Real ignition events come from the **NASA FIRMS** global 24-hour VIIRS C2 archive
(78,688 hotspots). Each cell is one of four states:

| State | Meaning          |
|-------|------------------|
| 0     | Non-burnable     |
| 1     | Susceptible      |
| 2     | Burning (active) |
| 3     | Burned / ash     |

Spread rule (Moore neighbourhood, 8 neighbours):

```
p_ignite = 1 − (1 − 0.4)^n_burning
```

A burning cell always transitions to burned after one step. Simulation runs for
20 steps with `numpy.random.seed(42)`.

### Parallelisation strategy

1D row decomposition via MPI. Each process owns `N/p` rows plus two ghost rows
exchanged with neighbours using `MPI.Sendrecv` at every step. After 20 steps,
`MPI.Gather` reassembles the full grid at rank 0.

### How to run

```bash
cd exercise_3
python fetch_firms_data.py          # download NASA FIRMS hotspots
python serial_ca.py                 # serial simulation, saves snapshots
mpiexec -n 4 python parallel_ca.py  # parallel MPI run
```

### Results

| Processes | Time (s) | Speedup | Efficiency |
|-----------|----------|---------|------------|
| 1         | 0.3342   | 1.000   | 1.000      |
| 2         | 0.2699   | 1.238   | 0.619      |
| 4         | 0.1561   | 2.141   | 0.535      |
| 8         | 0.0625   | 5.347   | 0.668      |

**5.3× speedup at 8 processes.** The non-monotonic efficiency (dip at p=4,
recovery at p=8) reflects the interplay between ghost-row communication latency
and per-process compute savings as the local row count shrinks.

---

## Exercise 4 — Parallel K-Means Clustering

### Context

K-Means clustering (K=7) on the **UCI Covertype** dataset:
581,012 records × 54 continuous features, Z-score normalised.
K=7 matches the number of natural cover types in the dataset.

### Parallelisation strategy

Data parallelism via MPI. Root scatters rows with `MPI.Scatterv`; each process
computes local distances and cluster assignments. Local sums and counts are
reduced globally with `MPI.Allreduce(SUM)` so all ranks update centroids
identically — no extra broadcast needed. Convergence tolerance: 10⁻⁴.

### How to run

```bash
cd exercise_4
python fetch_covertype.py           # download and preprocess dataset
python serial_kmeans.py             # serial baseline
mpiexec -n 4 python parallel_kmeans.py  # parallel MPI run
```

### Results

| Processes | Time (s) | Speedup | Efficiency |
|-----------|----------|---------|------------|
| 1         | 10.7802  | 1.000   | 1.000      |
| 2         |  5.8831  | 1.832   | 0.916      |
| 4         |  3.1860  | 3.383   | 0.845      |
| 8         |  2.3300  | 4.626   | 0.578      |

**Near-linear speedup up to 4 processes (84.5 % efficiency).** Efficiency drops
at p=8 as the two `Allreduce` calls per iteration become a fixed communication
cost relative to the shrinking per-process compute (~72,000 rows), consistent
with Amdahl's Law.

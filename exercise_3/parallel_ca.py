import numpy as np
import time
import pandas as pd
from mpi4py import MPI
import os

def get_grid(N, M, df):
    grid = np.ones((N, M), dtype=np.int8)
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    
    if lat_max == lat_min: lat_max += 0.1
    if lon_max == lon_min: lon_max += 0.1
        
    for _, row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        i = int((lat - lat_min) / (lat_max - lat_min) * (N - 1))
        j = int((lon - lon_min) / (lon_max - lon_min) * (M - 1))
        grid[i, j] = 2
    return grid

def local_step(local_grid_with_ghost, random_state):
    padded = np.pad(local_grid_with_ghost, ((0, 0), (1, 1)), mode='constant', constant_values=0)
    
    N_inner = local_grid_with_ghost.shape[0] - 2
    M_inner = local_grid_with_ghost.shape[1]
    
    burning = (padded == 2).astype(np.int8)
    burning_neighbors = (
        burning[:-2, :-2] + burning[:-2, 1:-1] + burning[:-2, 2:] +
        burning[1:-1, :-2]                     + burning[1:-1, 2:] +
        burning[2:, :-2]  + burning[2:, 1:-1]  + burning[2:, 2:]
    )
    
    prob_ignite = 1.0 - (1.0 - 0.4)**burning_neighbors
    random_matrix = random_state.random((N_inner, M_inner))
    
    inner_grid = local_grid_with_ghost[1:-1, :]
    new_inner = inner_grid.copy()
    
    ignite = (inner_grid == 1) & (random_matrix < prob_ignite)
    new_inner[ignite] = 2
    new_inner[inner_grid == 2] = 3
    
    new_local = local_grid_with_ghost.copy()
    new_local[1:-1, :] = new_inner
    return new_local

def save_snapshot(grid, step_num):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['gray', 'green', 'red', 'black'])
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=3)
    plt.title(f"Parallel Fire Spread - Step {step_num}")
    plt.axis('off')
    os.makedirs('../docs/assets', exist_ok=True)
    plt.savefig(f'../docs/assets/parallel_ca_step_{step_num}.png')
    plt.close()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Fix seed per rank to ensure reproducibility and realistic randomness
    random_state = np.random.RandomState(42 + rank) 
    
    N, M = 800, 800
    steps = 20
    
    if N % size != 0:
        if rank == 0:
            print("Error: N must be divisible by the number of processes.")
        comm.Abort()
        
    N_local = N // size
    local_grid = np.zeros((N_local + 2, M), dtype=np.int8)
    
    if rank == 0:
        try:
            df = pd.read_csv('hotspots.csv')
            global_grid = get_grid(N, M, df)
            print(f"Starting Parallel CA. Grid: {N}x{M}, Steps: {steps}, Procs: {size}")
        except FileNotFoundError:
            print("Error: hotspots.csv not found. Run fetch_firms_data.py first!")
            comm.Abort()
    else:
        global_grid = None

    recvbuf = np.zeros((N_local, M), dtype=np.int8)
    comm.Scatter(global_grid, recvbuf, root=0)
    
    local_grid[1:-1, :] = recvbuf
    
    comm.Barrier()
    start_time = time.perf_counter()
    
    for s in range(1, steps + 1):
        # Top boundary exchange
        if rank > 0:
            comm.Sendrecv(local_grid[1, :], dest=rank-1, recvbuf=local_grid[0, :], source=rank-1)
        else:
            local_grid[0, :] = 0
            
        # Bottom boundary exchange
        if rank < size - 1:
            comm.Sendrecv(local_grid[-2, :], dest=rank+1, recvbuf=local_grid[-1, :], source=rank+1)
        else:
            local_grid[-1, :] = 0
            
        local_grid = local_step(local_grid, random_state)
        
    comm.Barrier()
    exec_time = time.perf_counter() - start_time
    
    gatherbuf = None
    if rank == 0:
        gatherbuf = np.zeros((N, M), dtype=np.int8)
    comm.Gather(local_grid[1:-1, :], gatherbuf, root=0)
    
    if rank == 0:
        print(f"[Parallel] Finished in {exec_time:.4f} seconds")
        save_snapshot(gatherbuf, steps)

if __name__ == '__main__':
    main()

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.colors import ListedColormap

def get_grid(N, M, df):
    # 0: Non-burnable, 1: Susceptible, 2: Burning, 3: Burned
    grid = np.ones((N, M), dtype=np.int8)
    lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
    lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
    
    # Avoid division by zero if all points have same coordinate
    if lat_max == lat_min: lat_max += 0.1
    if lon_max == lon_min: lon_max += 0.1
        
    for _, row in df.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        i = int((lat - lat_min) / (lat_max - lat_min) * (N - 1))
        j = int((lon - lon_min) / (lon_max - lon_min) * (M - 1))
        grid[i, j] = 2 # Set ignition points
    return grid

def step(grid):
    N, M = grid.shape
    new_grid = grid.copy()
    padded = np.pad(grid, 1, mode='constant', constant_values=0)
    
    burning = (padded == 2).astype(np.int8)
    burning_neighbors = (
        burning[:-2, :-2] + burning[:-2, 1:-1] + burning[:-2, 2:] +
        burning[1:-1, :-2]                     + burning[1:-1, 2:] +
        burning[2:, :-2]  + burning[2:, 1:-1]  + burning[2:, 2:]
    )
    
    # Probability of ignition increases with burning neighbors
    prob_ignite = 1.0 - (1.0 - 0.4)**burning_neighbors
    random_matrix = np.random.random((N, M))
    
    ignite = (grid == 1) & (random_matrix < prob_ignite)
    new_grid[ignite] = 2
    new_grid[grid == 2] = 3 # burning becomes burned
    
    return new_grid

def save_snapshot(grid, step_num):
    cmap = ListedColormap(['gray', 'green', 'red', 'black'])
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=3)
    plt.title(f"Serial Fire Spread - Step {step_num}")
    plt.axis('off')
    os.makedirs('../docs/assets', exist_ok=True)
    plt.savefig(f'../docs/assets/serial_ca_step_{step_num}.png')
    plt.close()

def main():
    np.random.seed(42) # Fix seed for reproducibility
    N, M = 800, 800
    steps = 20
    
    try:
        df = pd.read_csv('hotspots.csv')
    except FileNotFoundError:
        print("Error: hotspots.csv not found. Run fetch_firms_data.py first!")
        return

    grid = get_grid(N, M, df)
    
    print(f"Starting Serial CA. Grid: {N}x{M}, Steps: {steps}")
    
    save_snapshot(grid, 0) # Mover fuera del reloj
    
    start_time = time.perf_counter()
    for s in range(1, steps + 1):
        grid = step(grid)
            
    exec_time = time.perf_counter() - start_time
    print(f"[Serial] Finished in {exec_time:.4f} seconds")
    
    save_snapshot(grid, steps) # Mover fuera del reloj

if __name__ == '__main__':
    main()

import numpy as np
import pandas as pd
import time
from mpi4py import MPI

def euclidean_distance(X, centroids):
    x_sq = np.sum(X**2, axis=1, keepdims=True)
    c_sq = np.sum(centroids**2, axis=1)
    dist_sq = x_sq + c_sq - 2 * np.dot(X, centroids.T)
    dist_sq = np.maximum(dist_sq, 0)
    return np.sqrt(dist_sq)

def parallel_kmeans(local_X, centroids_init, K, D, comm, rank, size, max_iters=50, tol=1e-4):
    centroids = centroids_init.copy()
    
    # Broadcast initial centroids from root
    comm.Bcast(centroids, root=0)
    
    total_start_time = time.time()
    
    for iteration in range(max_iters):
        iter_start = time.time()
        
        # 1. Assignment step (local computation)
        distances = euclidean_distance(local_X, centroids)
        local_labels = np.argmin(distances, axis=1)
        
        # 2. Local accumulation
        local_sums = np.zeros((K, D), dtype=np.float32)
        local_counts = np.zeros(K, dtype=np.int32)
        
        for k in range(K):
            mask = (local_labels == k)
            local_counts[k] = np.sum(mask)
            if local_counts[k] > 0:
                local_sums[k] = np.sum(local_X[mask], axis=0)
                
        # 3. Global reduction
        global_sums = np.zeros_like(local_sums)
        global_counts = np.zeros_like(local_counts)
        
        comm.Allreduce(local_sums, global_sums, op=MPI.SUM)
        comm.Allreduce(local_counts, global_counts, op=MPI.SUM)
        
        # 4. Update centroids (all processes update identically)
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            if global_counts[k] > 0:
                new_centroids[k] = global_sums[k] / global_counts[k]
            else:
                new_centroids[k] = centroids[k] # Keep old if empty
                
        # 5. Check convergence
        diff = np.linalg.norm(new_centroids - centroids)
        
        centroids = new_centroids
        iter_time = time.time() - iter_start
        
        if rank == 0:
            print(f"Iter {iteration+1:02d} | Time: {iter_time:.4f}s | Centroid Diff: {diff:.6f}")
            
        if diff < tol:
            if rank == 0:
                print(f"Converged at iteration {iteration+1}")
            break
            
    total_time = time.time() - total_start_time
    if rank == 0:
        print(f"[Parallel] Total K-Means Time: {total_time:.4f} seconds with {size} processes")
        
    return centroids, local_labels

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    X = None
    K = 7
    D = 54
    
    if rank == 0:
        try:
            print("Loading dataset...")
            df = pd.read_csv('covtype_scaled.csv')
            X = df.values.astype(np.float32)
            N_global = X.shape[0]
            print(f"Starting Parallel K-Means. Dataset Shape: {X.shape}, Clusters: {K}, Procs: {size}")
        except FileNotFoundError:
            print("Error: covtype_scaled.csv not found. Run fetch_covertype.py first!")
            comm.Abort()
            return
            
        # Ensure exact same random initialization as serial baseline
        np.random.seed(42)
        initial_idx = np.random.choice(N_global, K, replace=False)
        centroids_init = X[initial_idx].copy()
    else:
        centroids_init = np.zeros((K, D), dtype=np.float32)
        N_global = None

    # Broadcast N_global to all processes
    N_global = comm.bcast(N_global, root=0)
    
    # Calculate partition sizes for Scatterv
    counts = [N_global // size + (1 if p < N_global % size else 0) for p in range(size)]
    displacements = [sum(counts[:p]) for p in range(size)]
    
    local_N = counts[rank]
    local_X = np.zeros((local_N, D), dtype=np.float32)
    
    if rank == 0:
        counts_flat = np.array(counts) * D
        displ_flat = np.array(displacements) * D
        sendbuf = [X, counts_flat, displ_flat, MPI.FLOAT]
    else:
        sendbuf = None
        
    # Distribute the dataset across processes
    comm.Scatterv(sendbuf, [local_X, MPI.FLOAT], root=0)
    
    parallel_kmeans(local_X, centroids_init, K, D, comm, rank, size, tol=1e-4)

if __name__ == '__main__':
    main()

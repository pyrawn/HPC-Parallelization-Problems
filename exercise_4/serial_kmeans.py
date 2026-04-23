import numpy as np
import pandas as pd
import time

def euclidean_distance(X, centroids):
    # Vectorized distance calculation: (X - C)^2 = X^2 + C^2 - 2XC
    x_sq = np.sum(X**2, axis=1, keepdims=True)
    c_sq = np.sum(centroids**2, axis=1)
    dist_sq = x_sq + c_sq - 2 * np.dot(X, centroids.T)
    dist_sq = np.maximum(dist_sq, 0)
    return np.sqrt(dist_sq)

def serial_kmeans(X, K, max_iters=50, tol=1e-4):
    np.random.seed(42)
    N, D = X.shape
    
    # Initialize centroids randomly from data points
    initial_idx = np.random.choice(N, K, replace=False)
    centroids = X[initial_idx].copy()
    
    print(f"Starting Serial K-Means. Points: {N}, Dims: {D}, Clusters: {K}")
    start_time = time.time()
    
    for iteration in range(max_iters):
        iter_start = time.time()
        
        # 1. Assignment step
        distances = euclidean_distance(X, centroids)
        labels = np.argmin(distances, axis=1)
        
        # 2. Update step
        new_centroids = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centroids[k] = cluster_points.mean(axis=0)
            else:
                new_centroids[k] = centroids[k] # Keep unchanged if cluster is empty
                
        # 3. Check convergence
        diff = np.linalg.norm(new_centroids - centroids)
        iter_time = time.time() - iter_start
        print(f"Iter {iteration+1:02d} | Time: {iter_time:.4f}s | Centroid Diff: {diff:.6f}")
        
        centroids = new_centroids
        if diff < tol:
            print(f"Converged at iteration {iteration+1}")
            break
            
    total_time = time.time() - start_time
    print(f"[Serial] Total K-Means Time: {total_time:.4f} seconds")
    return centroids, labels

def main():
    try:
        print("Loading dataset...")
        df = pd.read_csv('covtype_scaled.csv')
        X = df.values.astype(np.float32) 
    except FileNotFoundError:
        print("Error: covtype_scaled.csv not found. Run fetch_covertype.py first!")
        return
        
    K = 7 # Covertype dataset naturally has 7 cover types
    serial_kmeans(X, K)

if __name__ == '__main__':
    main()

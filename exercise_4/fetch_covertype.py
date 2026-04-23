import pandas as pd
import requests
import os

def download_and_preprocess(output_file='covtype_scaled.csv'):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    gz_file = "covtype.data.gz"
    
    if not os.path.exists(output_file):
        print(f"Downloading Covertype dataset from {url}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(gz_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print("Extracting and loading into pandas...")
        # Header is not present in the dataset
        df = pd.read_csv(gz_file, header=None)
        
        # We drop the target variable (column 54) for unsupervised clustering (K-Means)
        X = df.iloc[:, :-1].values
        
        # Standardization (Z-score)
        print("Preprocessing (Z-score normalization)...")
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1 # Prevent division by zero for constant features
        X_scaled = (X - mean) / std
        
        df_scaled = pd.DataFrame(X_scaled)
        df_scaled.to_csv(output_file, index=False)
        print(f"Preprocessed dataset saved to {output_file} (Shape: {X_scaled.shape})")
        
        os.remove(gz_file)
    else:
        print(f"{output_file} already exists. Skipping download.")

if __name__ == '__main__':
    download_and_preprocess()

import os, gc, gzip, pickle, tempfile
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

def pca_fit(data, sample_size):
    # Calc full PCA
    # Create a generator for reproducibility
    rng = np.random.default_rng()
    # For a 2D array `arr` with shape (N, M)
    sampled_rows = rng.choice(data.shape[0], size=sample_size, replace=False)  # Indices
    X = data[sampled_rows]

    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Full PCA 
    pca = PCA(n_components=None)
    pca.fit(X_scaled)
    return scaler, pca

def pca_transform(B, scaler, pca_mean, X):
    """Transform data into PCA components space.
    
    Args:
        B: PCA components matrix (M x M) from pca_analysis
        scaler: StandardScaler object from pca_analysis
        pca_mean: Mean vector from PCA analysis (scaler.mean_ is different!)
        X: New data (N x M) to transform
    
    Returns:
        Y: Transformed data in PCA space (N x M)
    """
    # Standardize using original scaler
    X_scaled = scaler.transform(X)
    
    # Center using PCA mean (critical for correct projection)
    X_centered = X_scaled - pca_mean
    
    # Project onto PCA components
    Y = X_centered @ B
    
    return Y

def pca_transform_batch_export_dealloc(B, scaler, pca_mean, X, batch_size, fn):
    """Process X in batches, write compressed chunks, then combine results.
    
    Args:
        B: PCA components matrix (M x M)
        scaler: Fitted StandardScaler object
        pca_mean: PCA mean vector (M,)
        X: Input data (N x M)
        batch_size: Number of samples per batch
        fn: Final output filename (.npz)
    """
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    chunk_paths = []
    
    try:
        # Process and export batches
        for i in tqdm(range(0, len(X), batch_size), total=int(len(X)/batch_size)):
            batch = X[i:i + batch_size]
            
            # Transform batch
            X_scaled = scaler.transform(batch)
            X_centered = X_scaled - pca_mean
            Y_batch = X_centered @ B
            
            # Export compressed chunk
            chunk_path = os.path.join(temp_dir, f'chunk_{i}.pkl.gz')
            with gzip.open(chunk_path, 'wb') as f:
                pickle.dump(Y_batch, f)
            chunk_paths.append(chunk_path)
            
        # Clear memory
        del X
        gc.collect()
        
        # Reassemble chunks
        Y_chunks = []
        for path in tqdm(chunk_paths):
            with gzip.open(path, 'rb') as f:
                Y_chunks.append(pickle.load(f))
        
        Y = np.vstack(Y_chunks)
        
        # Export final result
        np.savez_compressed(fn, Y=Y)
        
    finally:
        # Cleanup
        for path in chunk_paths:
            try:
                os.remove(path)
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass

def pca_inverse_transform(B, scaler, pca_mean, Y):
    """Reconstruct data from PCA components.
    
    Args:
        B: PCA components matrix (M x M) from pca_analysis
        scaler: StandardScaler object from pca_analysis
        pca_mean: Mean vector from PCA analysis
        Y: Transformed data (N x K) where K ≤ M
    
    Returns:
        X_recon: Reconstructed data in original space (N x M)
    """
    # Handle partial component reconstruction
    if Y.shape[1] < B.shape[1]:
        B_partial = B[:, :Y.shape[1]]
    else:
        B_partial = B
    
    # Reconstruct in centered space
    X_centered = Y @ B_partial.T
    
    # Uncenter and inverse standardize
    X_scaled = X_centered + pca_mean
    X_recon = scaler.inverse_transform(X_scaled)
    
    return X_recon


if __name__=="__main__":
    # Generate random data
    X = np.random.randn(100, 5)  # 100 samples, 5 features
    
    # Run analysis
    B, scaler, L, V, MSE = pca_analysis(X)
    print("B", B)
    print("L", L)
    print("V", V)
    print("mse", MSE)
    # Results:
    # B.shape -> (5, 5)
    # len(L) -> 5
    # V[-1] -> 100.0 (last value)
    # len(MSE) -> 5

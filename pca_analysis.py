import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

def pca_analysis(X):
    # Standardize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Full PCA (compute ONCE)
    pca = PCA(n_components=None)
    pca.fit(X_scaled)
    
    # Get basis and eigenvalues
    B = pca.components_.T  # M x M basis
    L = pca.explained_variance_
    V = np.cumsum(pca.explained_variance_ratio_) * 100
    
    # Precompute centered data
    X_centered = X_scaled - pca.mean_
    
    MSE = np.zeros(X.shape[1])
    n_components = min(400, X.shape[1])                 # if it's more than 400, forget about it
    for i in tqdm(range(1, n_components+1)):
        # Manual projection and reconstruction
        proj = X_centered @ B[:, :i]  # Project to i components
        recon = proj @ B[:, :i].T     # Reconstruct in centered space
        
        # Uncenter and inverse standardize
        X_scaled_hat_i = recon + pca.mean_
        X_hat_i = scaler.inverse_transform(X_scaled_hat_i)
        
        # Calculate MSE
        MSE[i-1] = mean_squared_error(X, X_hat_i)
    
    return B, scaler, L, V, MSE, pca.mean_

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

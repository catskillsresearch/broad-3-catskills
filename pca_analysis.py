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
    for i in tqdm(range(1, X.shape[1]+1)):
        # Manual projection and reconstruction
        proj = X_centered @ B[:, :i]  # Project to i components
        recon = proj @ B[:, :i].T     # Reconstruct in centered space
        
        # Uncenter and inverse standardize
        X_scaled_hat_i = recon + pca.mean_
        X_hat_i = scaler.inverse_transform(X_scaled_hat_i)
        
        # Calculate MSE
        MSE[i-1] = mean_squared_error(X, X_hat_i)
    
    return B, scaler, L, V, MSE, pca.mean_

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

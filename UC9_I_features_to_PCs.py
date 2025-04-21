import os, luigi
from UC9_I_patches_to_features import UC9_I_patches_to_features
from read_assets_from_h5 import read_assets_from_h5
import joblib
import matplotlib.pyplot as plt
from pca_analysis import pca_analysis
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
        
class UC9_I_features_to_PCs(luigi.Task):
    def requires(self):
        return {'features': UC9_I_patches_to_features() }
        
    def output(self):
        return {
            'PCs': luigi.LocalTarget('resources/run/UC9_I_feature_PCs.npz'),
            'MSE': luigi.LocalTarget('resources/run/UC9_I_features_PCA_MSE.png'),
            'feature_scaler': luigi.LocalTarget('resources/run/UC9_I_feature_PCs_scaler.joblib'),
            'feature_pca_mean': luigi.LocalTarget('resources/run/UC9_I_feature_pca_mean.npz'),
            'feature_density': luigi.LocalTarget('resources/run/UC9_I_feature_density.png'),
            'feature_explained_variance': luigi.LocalTarget('resources/run/UC9_I_pca_basis_explained_variance.npz'),
            'feature_MSE': luigi.LocalTarget('resources/run/UC9_I_pca_basis_MSE.npz') }

    def run(self):
        assets, _ = read_assets_from_h5('resources/run/UC9_I_features.h5')
        features = assets['embeddings']
        index = assets['barcodes']
        plt.hist(features.flatten(), bins=100, density=True)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title("Chip Resnet50 Feature density")
        plt.savefig('resources/run/UC9_I_feature_density.png', dpi=150, bbox_inches='tight')
        B, scaler, L, V, MSE, pca_mean = pca_analysis(features[::32])
        joblib.dump(scaler, 'resources/run/UC9_I_feature_PCs_scaler.joblib')
        np.savez_compressed(f'resources/run/UC9_I_feature_pca_mean.npz', pca_mean)
        np.savez_compressed('resources/run/UC9_I_pca_basis_explained_variance.npz', V)
        np.savez_compressed('resources/run/UC9_I_pca_basis_MSE.npz', MSE)
        mse_goal = 0.16
        finish = np.where((MSE <=mse_goal))[-1][0]
        plt.plot(MSE)
        plt.xlim([finish-20, finish+20])
        plt.ylim([0.12,0.20])
        plt.scatter([finish],[mse_goal],color='red', s=40)
        plt.title(f'Use {finish} PCs for goal of reconstruction MSE <= 0.16')
        plt.savefig('resources/run/UC9_I_features_PCA_MSE.png', dpi=150, bbox_inches='tight')
        basis = B[:, :finish]
        X_scaled = scaler.fit_transform(features)
        X_centered = X_scaled - pca_mean
        PCs = X_centered @ basis
        np.savez_compressed(f'resources/run/UC9_I_feature_PCs.npz', PCs)

if __name__ == "__main__":
    luigi.build(
        [UC9_I_features_to_PCs()],  # Replace with your task class
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

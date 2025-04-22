import os, luigi
from UC9_I_patches_to_features import UC9_I_patches_to_features
from read_assets_from_h5 import read_assets_from_h5
import joblib
import matplotlib.pyplot as plt
from pca_analysis import pca_analysis
import numpy as np
        
class UC9_I_object_to_PCs(luigi.Task):
    object_type = luigi.Parameter()
    dependency_task = luigi.TaskParameter()  # Takes Task class as parameter
    
    def requires(self):
        return dependency_task()
        
    def output(self):
        return {
            'PCs': luigi.LocalTarget(f'resources/run/UC9_I_{self.object_type}_PCs.npz'),
            'MSE': luigi.LocalTarget(f'resources/run/UC9_I_{self.object_type}_PCA_MSE.png'),
            'scaler': luigi.LocalTarget(f'resources/run/UC9_I_{self.object_type}_PCs_scaler.joblib'),
            'pca_mean': luigi.LocalTarget(f'resources/run/UC9_I_{self.object_type}_pca_mean.npz'),
            'density': luigi.LocalTarget(f'resources/run/UC9_I_{self.object_type}_density.png'),
            'explained_variance': luigi.LocalTarget(f'resources/run/UC9_I_{self.object_type}_pca_basis_explained_var.npz'),
            'mse': luigi.LocalTarget(f'resources/run/UC9_I_{self.object_type}_pca_basis_MSE.npz') }

    def run(self):
        object_fn = 
        assets, _ = read_assets_from_h5(self.input().path)
        features = assets['embeddings']
        plt.hist(features.flatten(), bins=100, density=True)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f"{self.object_type} density")
        out = self.output()
        plt.savefig(out['density'].path, dpi=150, bbox_inches='tight')
        B, scaler, L, V, MSE, pca_mean = pca_analysis(features[::32])
        joblib.dump(scaler, out['scaler'].path)
        np.savez_compressed(out['pca_mean'].path, pca_mean)
        np.savez_compressed(out['explained_variance'].path, V)
        np.savez_compressed(out['mse'].path, MSE)
        mse_goal = 0.16
        finish = np.where((MSE <=mse_goal))[-1][0]
        plt.plot(MSE)
        plt.xlim([finish-20, finish+20])
        plt.ylim([0.12,0.20])
        plt.scatter([finish],[mse_goal],color='red', s=40)
        plt.title(f'Use {finish} PCs for {self.object_type} reconstruction MSE <= 0.16')
        plt.savefig(out['MSE'].path, dpi=150, bbox_inches='tight')
        basis = B[:, :finish]
        X_scaled = scaler.fit_transform(features)
        X_centered = X_scaled - pca_mean
        PCs = X_centered @ basis
        np.savez_compressed(out['PCs'].path, PCs)
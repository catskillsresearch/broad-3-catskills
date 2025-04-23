import os, luigi
from read_assets_from_h5 import read_assets_from_h5
import joblib
import matplotlib.pyplot as plt
from pca_analysis import pca_analysis
import numpy as np
        
class template_pca_fit_transform(luigi.Task):
    object_type = luigi.Parameter()
    object_name = luigi.Parameter()
    mse_goal = luigi.FloatParameter()
    dependency_task = luigi.TaskParameter()  # Takes Task class as parameter
    
    def requires(self):
        return self.dependency_task()
        
    def output(self):
        return {
            'scaler': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_PCs_scaler.joblib'),
            'pca_mean': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_pca_mean.npz'),
            'basis': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_pca_basis.npz'),
            'PCs': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_PCs.npz'),
            'MSE': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_PCA_MSE.png'),
            'density': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_density.png'),
            'explained_variance': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_pca_basis_explained_var.npz'),
            'mse': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_pca_basis_MSE.npz') }

    def run(self):
        out = self.output()
        
        data = np.load(self.input().path, allow_pickle=True)['arr_0']
        data_flat = data.flatten()
        data_flat = data_flat[data_flat != 0]
        plt.hist(data_flat, bins=100, density=True)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f"Non-0 {self.object_type} density")
        plt.savefig(out['density'].path, dpi=150, bbox_inches='tight')
        plt.clf()
        
        # Create a generator for reproducibility
        rng = np.random.default_rng()
        # For a 2D array `arr` with shape (N, M)
        sampled_rows = rng.choice(data.shape[0], size=10000, replace=False)  # Indices
        sample = data[sampled_rows]  # Subset rows

        B, scaler, L, V, MSE, pca_mean = pca_analysis(sample)
        joblib.dump(scaler, out['scaler'].path)
        np.savez_compressed(out['pca_mean'].path, pca_mean)
        np.savez_compressed(out['explained_variance'].path, V)
        np.savez_compressed(out['mse'].path, MSE)
        finish = np.where((MSE <= self.mse_goal))[-1][0]
        plt.plot(MSE)
        plt.xlim([finish-20, finish+20])
        plt.ylim([self.mse_goal * 0.8, self.mse_goal * 1.2])
        plt.scatter([finish],[self.mse_goal],color='red', s=40)
        plt.title(f'Use {finish} PCs for {self.object_type} reconstruction MSE <= {self.mse_goal}')
        plt.savefig(out['MSE'].path, dpi=150, bbox_inches='tight')
        plt.clf()
        basis = B[:, :finish]
        X_scaled = scaler.fit_transform(data)
        X_centered = X_scaled - pca_mean
        PCs = X_centered @ basis
        np.savez_compressed(out['PCs'].path, PCs)
        np.savez_compressed(out['basis'].path, basis)
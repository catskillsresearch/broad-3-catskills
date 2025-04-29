import os, luigi
from read_assets_from_h5 import read_assets_from_h5
import joblib
import matplotlib.pyplot as plt
from pca_analysis import *
import numpy as np
from select_random_from_2D_array import select_random_from_2D_array

class template_pca_fit_transform(luigi.Task):
    object_type = luigi.Parameter()
    object_name = luigi.Parameter()
    mse_goal = luigi.FloatParameter()
    dependency_task = luigi.TaskParameter()  
    sub_input = luigi.Parameter()
    sample_size = luigi.IntParameter()
    
    def requires(self):
        return self.dependency_task()
        
    def output(self):
        return {
            'input': self.input(),
            'scaler': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_PCs_scaler.joblib'),
            'pca_mean': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_pca_mean.npz'),
            'basis': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_pca_basis.npz'),
            'PCs': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_PCs.npz'),
            'MSE': luigi.LocalTarget(f'mermaid/{self.object_name}_{self.object_type}_PCA_MSE.png'),
            'density': luigi.LocalTarget(f'mermaid/{self.object_name}_{self.object_type}_density.png'),
            'explained_variance': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_pca_basis_explained_var.npz'),
            'mse': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_pca_basis_MSE.npz') }

    def show_density(self, data):
        data_flat = select_random_from_2D_array(data, 1000)
        plt.hist(data_flat, bins=100, density=True)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f"Non-0 {self.object_type} density")
        out = self.output()
        plt.savefig(out['density'].path, dpi=150, bbox_inches='tight')
        plt.clf()
        
    def run(self):
        src = self.input()
        if self.sub_input != "":
            src = src[self.sub_input]
        data = np.load(src.path, allow_pickle=True)
        keys = [x for x in data]
        data = data[keys[0]]
        self.show_density(data)
        out = self.output()
        # Create a generator for reproducibility
        rng = np.random.default_rng()
        # For a 2D array `arr` with shape (N, M)
        sampled_rows = rng.choice(data.shape[0], size=self.sample_size, replace=False)  # Indices
        sample = data[sampled_rows]
        basis, scaler, L, V, MSE, pca_mean, finish = pca_analysis(sample, self.mse_goal, start=48, end = 50)
        print("got B", basis.size)
        np.savez_compressed(out['basis'].path, basis)
        joblib.dump(scaler, out['scaler'].path)
        np.savez_compressed(out['pca_mean'].path, pca_mean)
        np.savez_compressed(out['explained_variance'].path, V)
        np.savez_compressed(out['mse'].path, MSE)
        print("pictures")
        plt.xlim([finish-20, finish+20])
        plt.ylim([self.mse_goal * 0.8, self.mse_goal * 1.2])
        plt.scatter([finish],[self.mse_goal],color='red', s=40)
        plt.title(f'Use {finish} PCs for {self.object_type} reconstruction MSE <= {self.mse_goal}')
        plt.savefig(out['MSE'].path, dpi=150, bbox_inches='tight')
        plt.clf()
        print("inverse start")
        print("sampled_rows", data.shape)
        print("HAIL MARY")
        batch_size = 1000
        PCs = pca_transform_batch_export_dealloc(basis, scaler, pca_mean, data, batch_size, out['PCs'].path)
        print("done")

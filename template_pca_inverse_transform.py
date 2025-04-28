import os, luigi
from read_assets_from_h5 import read_assets_from_h5
import joblib
import matplotlib.pyplot as plt
from pca_analysis import pca_transform, pca_inverse_transform
from sklearn.metrics import mean_squared_error
import numpy as np
from select_random_from_2D_array import select_random_from_2D_array
from np_loadz import np_loadz

class template_pca_inverse_transform(luigi.Task):
    object_type = luigi.Parameter()
    object_name = luigi.Parameter()
    pca_fit_transform = luigi.TaskParameter() 
    source = luigi.TaskParameter() 
    source_field = luigi.Parameter()
    
    def requires(self):
        return {'fit': self.pca_fit_transform(),
                'source': self.source()}
        
    def output(self):
        return {
            self.object_type: luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}.npz'),
            'density_comparison': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_density.png') }

    def compare_densities(self, X_original, X):
        X_flat = select_random_from_2D_array(X, 10000)
        X_original_flat = select_random_from_2D_array(X_original, 10000)
        
        plt.hist(X_flat, bins=100, density=True, label='X')
        plt.hist(X_original_flat, bins=100, density=True, label='X(original)')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.title(f"Non-0 {self.object_type} densities for fitted and application data")
        plt.savefig(self.output()['density_comparison'].path, dpi=150, bbox_inches='tight')
        plt.clf()
      
    def run(self):
        fit = self.pca_fit_transform().output()
        B = np_loadz(fit['basis'].path)
        pca_mean = np_loadz(fit['pca_mean'].path)
        scaler = joblib.load(fit['scaler'].path)
        try:
            src_fn = self.input()['source'].path
            Y = np_loadz(src_fn)
        except:
            src_fn = self.input()['source'][self.source_field].path
            Y = np_loadz(src_fn)
        pca_src = fit['PCs'].path
        Y_original = np_loadz(pca_src)
        self.compare_densities(Y_original, Y)
        X = pca_inverse_transform(B, scaler, pca_mean, Y)
        X = np.maximum(0, X)  # Gene expressions are non-negative
        np.savez_compressed(self.output()[self.object_type].path, X)


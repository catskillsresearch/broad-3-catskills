import os, luigi
from read_assets_from_h5 import read_assets_from_h5
import joblib
import matplotlib.pyplot as plt
from pca_analysis import pca_transform, pca_inverse_transform
from sklearn.metrics import mean_squared_error
import numpy as np
        
class template_pca_transform(luigi.Task):
    object_type = luigi.Parameter()
    object_name = luigi.Parameter()
    mse_goal = luigi.FloatParameter()
    pca_fit_transform = luigi.TaskParameter() 
    source = luigi.TaskParameter() 
    
    def requires(self):
        return {'fit': self.pca_fit_transform(),
                'source': self.source()} 
        
    def output(self):
        return {
            'PCs': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_PCs.npz'),
            'source_MSE': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_PCA_MSE.png'),
            'density_comparison': luigi.LocalTarget(f'resources/run/{self.object_name}_{self.object_type}_density.png') }

    def compare_densities(self, X_original, X):
        X_flat = X.flatten()
        X_flat = X_flat[X_flat != 0]

        X_original_flat = X_original.flatten()
        X_original_flat = X_original_flat[X_original_flat != 0]
        
        plt.hist(X_flat, bins=100, density=True, label='X')
        plt.hist(X_original_flat, bins=100, density=True, label='X(original)')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f"Non-0 {self.object_type} densities for fitted and application data")
        plt.savefig(out['density_comparison'].path, dpi=150, bbox_inches='tight')
        plt.clf()

    def mse_analysis(self, X, X_hat):
        mse = mean_squared_error(X, Xhat)
        with open(self.output()['source_MSE'].path,'w') as f:
            print(mse, file=f)
        plt.plot(X_hat[0], label='Xhat', color='green')
        plt.plot(X[0], label='X', color='blue')
        plt.title(f'{self.object_type} {self.object_name} preservation of X for one sample under PCA round-trip')
        plt.legend()
        plt.savefig(self.output()['source_MSE'].path, dpi=150, bbox_inches='tight')
        plt.clf()
        
    def run(self):
        fit = self.input()['fit']
        src = self.input()['source']
        scaler = fit['scaler']
        pca_mean = fit['pca_mean']
        
        out = self.output()
        X_original = np.load(fit.input().path, allow_pickle=True)['arr_0']
        X = np.load(src.input().path, allow_pickle=True)['arr_0']

        self.compare_densities(X_original, X)
        Y = pca_transform(B, scaler, pca_mean, X)
        np.savez_compressed(out['PCs'].path, Y)
        
        Xhat = pca_inverse_transform(B, scaler, pca_mean, Y)
        self.mse_analysis(X, Xhat)

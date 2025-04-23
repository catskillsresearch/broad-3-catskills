import os, luigi, torch
from ridge_regression import *
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

class template_ridge_fit(luigi.Task):
    src_object_type = luigi.Parameter()
    src_object_name = luigi.Parameter()
    src_task = luigi.TaskParameter()
    tgt_object_type = luigi.Parameter()
    tgt_object_name = luigi.Parameter()
    tgt_task = luigi.TaskParameter()
    
    def requires(self):
        return {'src': self.src_task(), 'tgt': self.tgt_task()}
        
    def output(self):
        map_name = f'{self.src_object_name}_{self.src_object_type}_{self.tgt_object_name}_{self.tgt_object_type}_ridge_fit'
        return {
            'W': luigi.LocalTarget(f'resources/run/{map_name}_W.npz'),
            'mse': luigi.LocalTarget(f'resources/run/{map_name}_mse.txt'),
            'prediction': luigi.LocalTarget(f'resources/run/{map_name}_pred.png'),
            'spearman': luigi.LocalTarget(f'resources/run/{map_name}_spearman.png')}

    def mse_metric(self):
        mse = mean_squared_error(self.Y, self.Y_hat)
        with open(self.output()['mse'].path,'w') as f:
            print(mse, file=f)
        plt.plot(self.Y_hat[0], label='Yhat', color='green')
        plt.plot(self.Y[0], label='Y', color='blue')
        plt.title(f'Predicted vs Actual example for {self.tgt_object_type} via fitted Ridge regression')
        plt.legend()
        plt.savefig(self.output()['prediction'].path, dpi=150, bbox_inches='tight')
        plt.clf()

    def spearman_metric(self):
        correlations = [spearmanr(a_row, b_row)[0] for a_row, b_row in zip(self.Y, self.Y_hat)]
        mu = np.mean(correlations)
        plt.hist(correlations, density=True, bins=100);
        plt.axvline(x=mu, color='r', linestyle='--', alpha=0.7)
        plt.title(f"""Spearman rank correlation density between predicted and actual
        Mean={mu:0.3f}""");
        plt.savefig(self.output()['spearman'].path, dpi=150, bbox_inches='tight')
        plt.clf()
        
    def run(self):
        # Regression
        X = np.load(self.input()['src']['PCs'].path)['arr_0']
        self.Y = np.load(self.input()['tgt']['PCs'].path)['arr_0']
        W = ridge_fit(X, self.Y)
        np.savez_compressed(self.output()['W'].path, W)

        # Metrics
        self.Y_hat = ridge_apply(X, W)
        self.mse_metric()
        self.spearman_metric()
    
if __name__ == "__main__":
    luigi.build(
        [template_ridge_fit()],
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

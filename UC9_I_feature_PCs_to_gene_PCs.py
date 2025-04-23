import os, luigi, torch
from ridge_regression import *
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from UC9_I_features_to_PCs import UC9_I_features_to_PCs
from UC9_I_genes458_to_PCs import UC9_I_genes458_to_PCs

class UC9_I_feature_PCs_to_gene_PCs(luigi.Task):
    
    def requires(self):
        return {'features': UC9_I_features_to_PCs(), 
                'genes458': UC9_I_genes458_to_PCs()}
        
    def output(self):
        return {
            'W': luigi.LocalTarget(f'resources/run/UC9_I_feature_PCs_to_gene_PCs_W.npz'),
            'mse': luigi.LocalTarget(f'resources/run/UC9_I_feature_PCs_to_gene_PCs_mse.txt'),
            'prediction': luigi.LocalTarget(f'resources/run/UC9_I_feature_PCs_to_gene_PCs_pred.png'),
            'spearman': luigi.LocalTarget(f'resources/run/UC9_I_feature_PCs_to_gene_PCs_spearman.png')}

    def mse_metric(self):
        mse = mean_squared_error(self.gene_PCs, self.Y_hat)
        with open(self.output()['mse'].path,'w') as f:
            print(mse, file=f)
        plt.plot(self.Y_hat[0], label='Yhat', color='green')
        plt.plot(self.gene_PCs[0], label='Y', color='blue')
        plt.title('Yhat[0] vs Y[0] for imputed gene PCs on trained Ridge regression')
        plt.legend()
        plt.savefig(self.output()['prediction'].path, dpi=150, bbox_inches='tight')
        plt.clf()

    def spearman_metric(self):
        correlations = [spearmanr(a_row, b_row)[0] for a_row, b_row in zip(self.gene_PCs, self.Y_hat)]
        mu = np.mean(correlations)
        plt.hist(correlations, density=True, bins=100);
        plt.axvline(x=mu, color='r', linestyle='--', alpha=0.7)
        plt.title(f"""Spearman rank correlation density between predicted and actual
        Mean={mu:0.3f}""");
        plt.savefig(self.output()['spearman'].path, dpi=150, bbox_inches='tight')
        plt.clf()
        
    def run(self):
        # Regression
        feature_PCs = np.load(self.input()['features']['PCs'].path)['arr_0']
        self.gene_PCs = np.load(self.input()['genes458']['PCs'].path)['arr_0']
        W = ridge_fit(feature_PCs, self.gene_PCs)
        np.savez_compressed(self.output()['W'].path, W)

        # Metrics
        self.Y_hat = ridge_apply(feature_PCs, W)
        self.mse_metric()
        self.spearman_metric()
    
if __name__ == "__main__":
    luigi.build(
        [UC9_I_feature_PCs_to_gene_PCs()],
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

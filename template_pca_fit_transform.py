import os, luigi
from read_assets_from_h5 import read_assets_from_h5
import joblib
import matplotlib.pyplot as plt
from pca_analysis import pca_fit, pca_transform_batch_export_dealloc
import numpy as np
from select_random_from_2D_array import select_random_from_2D_array
from np_loadz import np_loadz
from kneed import KneeLocator

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
        tag = f'{self.object_name}_{self.object_type}'
        return {
            'input': self.input(),
            'scaler': luigi.LocalTarget(f'resources/run/{tag}_PCs_scaler.joblib'),
            'pca_mean': luigi.LocalTarget(f'resources/run/{tag}_pca_mean.npz'),
            'basis': luigi.LocalTarget(f'resources/run/{tag}_pca_basis.npz'),
            'PCs': luigi.LocalTarget(f'resources/run/{tag}_PCs.npz'),
            'density': luigi.LocalTarget(f'mermaid/{tag}_density.png'),
            'correlation_matrix': luigi.LocalTarget(f'mermaid/{tag}_correlation_matrix.png'),
            'correlation_matrix_density': luigi.LocalTarget(f'mermaid/{tag}_correlation_density.png'),
            'elbow_method': luigi.LocalTarget(f'mermaid/{tag}_elbow_method.png'),
            'kaiser_method': luigi.LocalTarget(f'mermaid/{tag}_kaiser_method.png'),
            'cumvar_method': luigi.LocalTarget(f'mermaid/{tag}_cumvar_method.png')}

    def show_density(self, data):
        data_flat = select_random_from_2D_array(data, 2000)
        plt.hist(data_flat, bins=200, density=True)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f"Non-0 {self.name} density")
        plt.savefig(self.output()['density'].path, dpi=150, bbox_inches='tight')
        plt.clf()
        return data_flat

    def plot_correlation_matrix(self, data):
        tag = f'{self.object_name}_{self.object_type}'
        rho = np.corrcoef(data.T)
        plt.imshow(rho)
        plt.title(f'{tag} correlation')
        plt.savefig(self.output()['correlation_matrix'].path, dpi=150, bbox_inches='tight')
        return rho                

    def plot_correlation_matrix_density(self, data_flat, rho):
        rho[rho>= 0.999] = np.nan
        rhomin, rhomax = np.nanmin(rho), np.nanmax(rho)
        plt.hist(data_flat, bins=200, density=True)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f"Off-diagonal feature correlation density in [{rhomin:.2f},{rhomax:.2f}]")
        plt.savefig(self.output()['correlation_matrix_density'], dpi=150, bbox_inches='tight')

    def plot_elbow_method(self, pca, K_nee):
        plt.figure(figsize=(8,6))
        plt.plot(np.arange(1, len(pca.explained_variance_)+1), pca.explained_variance_, marker='o')
        plt.xlabel('Component number')
        plt.ylabel('Eigenvalue')
        plt.title(f"{self.object_type} Scree components to elbow: {K_nee}")
        plt.axvline(K_nee, color='red', linestyle='--')  # Dashed red line
        plt.grid(True)
        plt.savefig(self.output()['elbow_method'].path, dpi=150, bbox_inches='tight')
        
    def elbow_method(self, pca):
        # Plot the eigenvalues and look for the "elbow" point where the marginal gain drops off. 
        K_nee = KneeLocator(range(1, len(pca.explained_variance_)+1), pca.explained_variance_, curve='convex', direction='decreasing').knee
        self.plot_elbow_method(pca, K_nee)
        return K_nee

    def plot_kaiser_method(self, pca, K_aiser):
        plt.figure(figsize=(8,6))
        plt.plot(np.arange(1, len(pca.explained_variance_)+1), pca.explained_variance_, marker='o')
        plt.xlabel('Component number')
        plt.ylabel('Eigenvalue')
        plt.title(f"{self.object_type} Kaiser components to eigenvalue < 1: {K_aiser}")
        plt.axvline(K_nee, color='red', linestyle='--')  # Dashed red line
        plt.grid(True)
        plt.savefig(self.output()['kaiser_method'].path, dpi=150, bbox_inches='tight')

    def kaiser_method(self, pca):
        # Keep components with eigenvalues > 1. Assumes standardized data.
        eigenvalues = pca.explained_variance_
        K_aiser = np.sum(eigenvalues > 1)
        self.plot_kaiser_method(pca, K_aiser)
        return K_aiser

    def plot_cumvar_method(self, pca, K_cumvar)
        V = np.cumsum(pca.explained_variance_ratio_)
        plt.figure(figsize=(8,6))
        plt.plot(V)
        plt.xlabel('Number of components')
        plt.ylabel(f"{self.object_type} cumulative explained variance")
        plt.title(f"{self.object_type} number of components to retain {goal*100:.0f}% variance: {K_cumvar}")
        plt.axvline(K_cumvar, color='red', linestyle='--')  # Dashed red line
        plt.plot(K_cumvar, goal, 'ko', markersize=8)  # Black dot at intersection
        plt.grid(True)
        plt.savefig(self.output()['cumvar_method'].path, dpi=150, bbox_inches='tight')
        
    def cumulative_variance_method(self, pca):
        explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        goal = 0.9
        K_cumvar = np.argmax(explained_variance_ratio >= goal) + 1
        self.plot_cumvar_method(pca, K_cumvar)
        return K_cumvar
        
    def assess_optimal_components(self, pca):
        K_nee = self.elbow_method(pca)
        K_aiser = self.kaiser_method(pca)
        K_cumvar = self.cumulative_variance_method(pca)
        K_final = max(2, min(K_cumvar, K_nee, K_aiser))
        return K_final

    def show_various_statistics(self, data):
        # Show various statistics for insight
        data_flat = self.show_density(data)
        rho = self.plot_correlation_matrix(data)
        self.plot_correlation_matrix_density(data_flat, rho)
        
    def run(self):
        src = self.input()
        if self.sub_input != "":
            src = src[self.sub_input]
        data = np_loadz(src.path)
        self.show_various_statistics(data)
        scaler, pca = pca_fit(data, self.sample_size)
        K = self.assess_optimal_components(pca)
        basis = pca.components_.T[:,:K]
        
        np.savez_compressed(out['basis'].path, basis)
        joblib.dump(scaler, out['scaler'].path)
        np.savez_compressed(out['pca_mean'].path, pca.mean_)

        batch_size = 1000
        PCs = pca_transform_batch_export_dealloc(basis, scaler, pca_mean, data, batch_size, self.output()['PCs'].path)

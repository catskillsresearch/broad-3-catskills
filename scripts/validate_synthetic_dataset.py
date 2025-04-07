#!/usr/bin/env python3
"""
Validation tests for synthetic dataset to verify statistical properties.
This script runs a series of tests to validate that the synthetic dataset
has the expected statistical properties and can be used for model evaluation.
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
import umap

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate synthetic dataset statistical properties')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing synthetic dataset')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for validation results')
    parser.add_argument('--visualize', action='store_true', help='Generate validation visualizations')
    return parser.parse_args()

def load_synthetic_dataset(dataset_dir):
    """
    Load synthetic dataset from directory.
    
    Parameters:
    -----------
    dataset_dir : str
        Directory containing synthetic dataset
        
    Returns:
    --------
    dataset : dict
        Dictionary containing dataset components
    """
    dataset = {}
    
    # Load gene expression data
    dataset['gene_expression'] = np.load(os.path.join(dataset_dir, 'gene_expression.npy'))
    
    # Load cell coordinates
    dataset['cell_coordinates'] = np.load(os.path.join(dataset_dir, 'cell_coordinates.npy'))
    
    # Load gene names
    dataset['gene_names'] = np.load(os.path.join(dataset_dir, 'gene_names.npy'))
    
    # Load region labels
    dataset['region_labels'] = np.load(os.path.join(dataset_dir, 'region_labels.npy'))
    
    # Load quality scores
    dataset['quality_scores'] = np.load(os.path.join(dataset_dir, 'quality_scores.npy'))
    
    # Load ground truth if available
    ground_truth_path = os.path.join(dataset_dir, 'ground_truth.npy')
    if os.path.exists(ground_truth_path):
        dataset['ground_truth'] = np.load(ground_truth_path, allow_pickle=True).item()
    
    # Load metadata
    with open(os.path.join(dataset_dir, 'metadata.yaml'), 'r') as f:
        dataset['metadata'] = yaml.safe_load(f)
    
    return dataset

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def test_distribution_properties(dataset, output_dir=None):
    """
    Test distribution properties of the synthetic dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dictionary containing dataset components
    output_dir : str
        Output directory for validation results
        
    Returns:
    --------
    results : dict
        Dictionary containing test results
    """
    print("Testing distribution properties...")
    
    gene_expression = dataset['gene_expression']
    n_cells, n_genes = gene_expression.shape
    
    results = {
        'distribution_tests': {}
    }
    
    # 1. Test for negative binomial distribution
    # For RNA-seq data, we expect a negative binomial distribution
    
    # Sample a few genes for testing
    sample_genes = np.random.choice(n_genes, min(10, n_genes), replace=False)
    
    for gene_idx in sample_genes:
        gene_expr = gene_expression[:, gene_idx]
        
        # Fit negative binomial distribution
        params = stats.nbinom.fit(gene_expr)
        
        # Perform Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.kstest(
            gene_expr, 
            lambda x: stats.nbinom.cdf(x, *params)
        )
        
        results['distribution_tests'][f'gene_{gene_idx}'] = {
            'ks_statistic': ks_statistic,
            'ks_pvalue': ks_pvalue,
            'nb_params': params
        }
        
        # Visualize if output directory is provided
        if output_dir:
            plt.figure(figsize=(10, 6))
            
            # Plot histogram of gene expression
            plt.hist(gene_expr, bins=30, density=True, alpha=0.7, label='Observed')
            
            # Plot fitted negative binomial distribution
            x = np.arange(0, np.max(gene_expr) + 1)
            y = stats.nbinom.pmf(x, *params)
            plt.plot(x, y, 'r-', label=f'Negative Binomial Fit (p={ks_pvalue:.4f})')
            
            plt.title(f'Gene {gene_idx} Expression Distribution')
            plt.xlabel('Expression Level')
            plt.ylabel('Density')
            plt.legend()
            
            plt.savefig(os.path.join(output_dir, f'distribution_gene_{gene_idx}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    # 2. Test for overall expression distribution
    # Calculate mean and variance for each gene
    gene_means = np.mean(gene_expression, axis=0)
    gene_vars = np.var(gene_expression, axis=0)
    
    # Calculate mean-variance relationship
    slope, intercept, r_value, p_value, std_err = stats.linregress(gene_means, gene_vars)
    
    results['mean_variance_relationship'] = {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err
    }
    
    # Visualize mean-variance relationship
    if output_dir:
        plt.figure(figsize=(10, 6))
        plt.scatter(gene_means, gene_vars, alpha=0.5)
        plt.plot(gene_means, intercept + slope * gene_means, 'r-', 
                 label=f'Linear Fit (r={r_value:.4f}, p={p_value:.4f})')
        plt.title('Mean-Variance Relationship')
        plt.xlabel('Mean Expression')
        plt.ylabel('Expression Variance')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'mean_variance_relationship.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Distribution tests complete.")
    return results

def test_spatial_properties(dataset, output_dir=None):
    """
    Test spatial properties of the synthetic dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dictionary containing dataset components
    output_dir : str
        Output directory for validation results
        
    Returns:
    --------
    results : dict
        Dictionary containing test results
    """
    print("Testing spatial properties...")
    
    cell_coordinates = dataset['cell_coordinates']
    region_labels = dataset['region_labels']
    gene_expression = dataset['gene_expression']
    
    results = {
        'spatial_tests': {}
    }
    
    # 1. Test for spatial clustering of cells by region
    # Calculate silhouette score for region labels
    silhouette = silhouette_score(cell_coordinates, region_labels)
    
    results['spatial_tests']['region_silhouette'] = silhouette
    
    # 2. Test for spatial autocorrelation of gene expression
    # Sample a few genes for testing
    n_genes = gene_expression.shape[1]
    sample_genes = np.random.choice(n_genes, min(5, n_genes), replace=False)
    
    for gene_idx in sample_genes:
        gene_expr = gene_expression[:, gene_idx]
        
        # Calculate Moran's I for spatial autocorrelation
        # First, create a distance matrix
        n_cells = len(cell_coordinates)
        dist_matrix = np.zeros((n_cells, n_cells))
        
        for i in range(n_cells):
            for j in range(i+1, n_cells):
                dist = np.linalg.norm(cell_coordinates[i] - cell_coordinates[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        # Convert distances to weights (inverse distance)
        # Avoid division by zero by adding a small constant
        weights = 1 / (dist_matrix + 1e-10)
        np.fill_diagonal(weights, 0)  # No self-connections
        
        # Normalize weights
        row_sums = weights.sum(axis=1)
        weights = weights / row_sums[:, np.newaxis]
        
        # Calculate Moran's I
        z_expr = gene_expr - np.mean(gene_expr)
        numerator = np.sum(weights * np.outer(z_expr, z_expr))
        denominator = np.sum(z_expr**2)
        
        morans_i = (n_cells / np.sum(weights)) * (numerator / denominator)
        
        results['spatial_tests'][f'gene_{gene_idx}_morans_i'] = morans_i
        
        # Visualize spatial distribution of gene expression
        if output_dir:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(cell_coordinates[:, 0], cell_coordinates[:, 1], 
                                 c=gene_expr, cmap='viridis', s=10, alpha=0.7)
            plt.colorbar(scatter, label='Expression')
            plt.title(f'Spatial Distribution of Gene {gene_idx} Expression (Moran\'s I = {morans_i:.4f})')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.savefig(os.path.join(output_dir, f'spatial_gene_{gene_idx}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    print("Spatial tests complete.")
    return results

def test_gene_module_properties(dataset, output_dir=None):
    """
    Test gene module properties of the synthetic dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dictionary containing dataset components
    output_dir : str
        Output directory for validation results
        
    Returns:
    --------
    results : dict
        Dictionary containing test results
    """
    print("Testing gene module properties...")
    
    gene_expression = dataset['gene_expression']
    gene_names = dataset['gene_names']
    
    # Check if ground truth is available
    has_ground_truth = 'ground_truth' in dataset and 'gene_modules' in dataset['ground_truth']
    
    results = {
        'module_tests': {}
    }
    
    # 1. Calculate gene-gene correlation matrix
    corr_matrix = np.corrcoef(gene_expression.T)
    
    # 2. If ground truth is available, compare with expected modules
    if has_ground_truth:
        gene_modules = dataset['ground_truth']['gene_modules']
        module_assignments = gene_modules['module_assignments']
        module_correlations = gene_modules['module_correlations']
        
        # For each module, calculate average correlation within the module
        for module_idx in range(len(module_correlations)):
            module_genes = np.where(module_assignments == module_idx)[0]
            
            if len(module_genes) <= 1:
                continue
            
            # Calculate average correlation within the module
            module_corr_matrix = corr_matrix[np.ix_(module_genes, module_genes)]
            avg_corr = (np.sum(module_corr_matrix) - len(module_genes)) / (len(module_genes) * (len(module_genes) - 1))
            
            # Compare with expected correlation
            expected_corr = module_correlations[module_idx]
            
            results['module_tests'][f'module_{module_idx}'] = {
                'expected_correlation': expected_corr,
                'observed_correlation': avg_corr,
                'difference': avg_corr - expected_corr,
                'n_genes': len(module_genes)
            }
            
            # Visualize module correlation matrix
            if output_dir:
                plt.figure(figsize=(10, 8))
                sns.heatmap(module_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0)
                plt.title(f'Module {module_idx} Correlation Matrix\nExpected: {expected_corr:.4f}, Observed: {avg_corr:.4f}')
                plt.savefig(os.path.join(output_dir, f'module_{module_idx}_correlation.png'), dpi=300, bbox_inches='tight')
                plt.close()
    
    # 3. Perform hierarchical clustering to identify modules
    from scipy.cluster.hierarchy import linkage, fcluster
    
    # Calculate linkage matrix
    linkage_matrix = linkage(corr_matrix, method='ward')
    
    # Determine optimal number of clusters
    from scipy.cluster.hierarchy import dendrogram
    
    if output_dir:
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, truncate_mode='lastp', p=30)
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Gene Index')
        plt.ylabel('Distance')
        plt.savefig(os.path.join(output_dir, 'gene_clustering_dendrogram.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Try different numbers of clusters
    n_clusters_range = range(2, 21)
    silhouette_scores = []
    
    for n_clusters in n_clusters_range:
        # Get cluster assignments
        cluster_assignments = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Calculate silhouette score
        silhouette = silhouette_score(corr_matrix, cluster_assignments)
        silhouette_scores.append(silhouette)
    
    # Find optimal number of clusters
    optimal_n_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    
    results['module_tests']['optimal_n_clusters'] = optimal_n_clusters
    results['module_tests']['max_silhouette_score'] = np.max(silhouette_scores)
    
    # Visualize silhouette scores
    if output_dir:
        plt.figure(figsize=(10, 6))
        plt.plot(n_clusters_range, silhouette_scores, 'o-')
        plt.axvline(x=optimal_n_clusters, color='r', linestyle='--', 
                   label=f'Optimal: {optimal_n_clusters} clusters')
        plt.title('Silhouette Score for Different Numbers of Gene Clusters')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'gene_clustering_silhouette.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Get cluster assignments for optimal number of clusters
    cluster_assignments = fcluster(linkage_matrix, optimal_n_clusters, criterion='maxclust')
    
    # If ground truth is available, compare with expected modules
    if has_ground_truth:
        # Calculate adjusted Rand index
        ari = adjusted_rand_score(module_assignments, cluster_assignments)
        results['module_tests']['adjusted_rand_index'] = ari
        
        # Visualize confusion matrix between ground truth and detected modules
        if output_dir:
            # Create confusion matrix
            confusion = np.zeros((np.max(module_assignments) + 1, np.max(cluster_assignments) + 1))
            
            for i in range(len(module_assignments)):
                confusion[module_assignments[i], cluster_assignments[i] - 1] += 1
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(confusion, cmap='Blues', annot=True, fmt='g')
            plt.title(f'Confusion Matrix: Ground Truth vs. Detected Modules (ARI = {ari:.4f})')
            plt.xlabel('Detected Module')
            plt.ylabel('Ground Truth Module')
            plt.savefig(os.path.join(output_dir, 'module_confusion_matrix.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    print("Gene module tests complete.")
    return results

def test_region_properties(dataset, output_dir=None):
    """
    Test region properties of the synthetic dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dictionary containing dataset components
    output_dir : str
        Output directory for validation results
        
    Returns:
    --------
    results : dict
        Dictionary containing test results
    """
    print("Testing region properties...")
    
    gene_expression = dataset['gene_expression']
    region_labels = dataset['region_labels']
    
    results = {
        'region_tests': {}
    }
    
    # 1. Test for differential expression between regions
    unique_regions = np.unique(region_labels)
    n_regions = len(unique_regions)
    n_genes = gene_expression.shape[1]
    
    # Calculate mean expression for each region
    region_means = np.zeros((n_regions, n_genes))
    
    for i, region in enumerate(unique_regions):
        region_cells = region_labels == region
        region_means[i] = np.mean(gene_expression[region_cells], axis=0)
    
    # Calculate variance of means across regions
    mean_variance = np.var(region_means, axis=0)
    
    # Identify top differentially expressed genes
    top_de_genes = np.argsort(mean_variance)[::-1][:20]
    
    results['region_tests']['top_de_genes'] = top_de_genes.tolist()
    results['region_tests']['mean_variance'] = mean_variance[top_de_genes].tolist()
    
    # 2. Test if regions can be identified from gene expression
    # Use PCA to reduce dimensionality
    pca = PCA(n_components=min(10, n_genes, gene_expression.shape[0]))
    pca_result = pca.fit_transform(gene_expression)
    
    # Train a classifier to predict region from gene expression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_scores = cross_val_score(clf, pca_result, region_labels, cv=5)
    
    results['region_tests']['region_prediction_cv_scores'] = cv_scores.tolist()
    results['region_tests']['region_prediction_mean_score'] = np.mean(cv_scores)
    
    # Visualize region-specific gene expression
    if output_dir:
        # Create heatmap of top DE genes by region
        plt.figure(figsize=(12, 10))
        
        # Prepare data for heatmap
        heatmap_data = np.zeros((len(top_de_genes), n_regions))
        
        for i, gene_idx in enumerate(top_de_genes):
            for j, region in enumerate(unique_regions):
                region_cells = region_labels == region
                heatmap_data[i, j] = np.mean(gene_expression[region_cells, gene_idx])
        
        # Scale data for better visualization
        heatmap_data = (heatmap_data - np.min(heatmap_data, axis=1, keepdims=True)) / \
                       (np.max(heatmap_data, axis=1, keepdims=True) - np.min(heatmap_data, axis=1, keepdims=True) + 1e-10)
        
        sns.heatmap(heatmap_data, cmap='viridis', 
                   xticklabels=[f'Region {r}' for r in unique_regions],
                   yticklabels=[f'Gene {g}' for g in top_de_genes])
        plt.title('Top Differentially Expressed Genes by Region')
        plt.xlabel('Region')
        plt.ylabel('Gene')
        plt.savefig(os.path.join(output_dir, 'region_de_genes.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Visualize PCA of gene expression colored by region
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=region_labels, cmap='tab10', s=10, alpha=0.7)
        plt.colorbar(scatter, label='Region')
        plt.title(f'PCA of Gene Expression Colored by Region (CV Score: {np.mean(cv_scores):.4f})')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.savefig(os.path.join(output_dir, 'region_pca.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Region tests complete.")
    return results

def validate_synthetic_dataset(dataset_dir, output_dir=None, visualize=False):
    """
    Validate synthetic dataset statistical properties.
    
    Parameters:
    -----------
    dataset_dir : str
        Directory containing synthetic dataset
    output_dir : str
        Output directory for validation results
    visualize : bool
        Whether to generate validation visualizations
        
    Returns:
    --------
    results : dict
        Dictionary containing validation results
    """
    print(f"Validating synthetic dataset in {dataset_dir}...")
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, 'validation')
    
    if visualize:
        ensure_dir(output_dir)
    
    # Load synthetic dataset
    dataset = load_synthetic_dataset(dataset_dir)
    
    # Run validation tests
    results = {}
    
    # Test distribution properties
    distribution_results = test_distribution_properties(dataset, output_dir if visualize else None)
    results.update(distribution_results)
    
    # Test spatial properties
    spatial_results = test_spatial_properties(dataset, output_dir if visualize else None)
    results.update(spatial_results)
    
    # Test gene module properties
    module_results = test_gene_module_properties(dataset, output_dir if visualize else None)
    results.update(module_results)
    
    # Test region properties
    region_results = test_region_properties(dataset, output_dir if visualize else None)
    results.update(region_results)
    
    # Save validation results
    if output_dir:
        results_file = os.path.join(output_dir, 'validation_results.yaml')
        
        # Convert numpy arrays to lists for YAML serialization
        results_serializable = {}
        
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj
        
        results_serializable = make_serializable(results)
        
        with open(results_file, 'w') as f:
            yaml.dump(results_serializable, f)
    
    print("Validation complete!")
    return results

def main():
    """Main function."""
    args = parse_args()
    validate_synthetic_dataset(args.dataset_dir, args.output_dir, args.visualize)

if __name__ == "__main__":
    main()

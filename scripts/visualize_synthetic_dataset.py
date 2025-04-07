#!/usr/bin/env python3
"""
Generate visualizations for synthetic dataset analysis and model evaluation.
This script creates a comprehensive set of visualizations to analyze synthetic
datasets and evaluate model performance against known ground truth.
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, UMAP
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate visualizations for synthetic dataset analysis')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Directory containing synthetic dataset')
    parser.add_argument('--predictions_dir', type=str, default=None, help='Directory containing model predictions')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for visualizations')
    parser.add_argument('--analysis_type', type=str, default='all', 
                        choices=['all', 'dataset', 'model', 'comparison'],
                        help='Type of analysis to perform')
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
    
    # Load indices
    dataset['train_indices'] = np.load(os.path.join(dataset_dir, 'train_indices.npy'))
    dataset['val_indices'] = np.load(os.path.join(dataset_dir, 'val_indices.npy'))
    dataset['test_indices'] = np.load(os.path.join(dataset_dir, 'test_indices.npy'))
    
    # Load ground truth if available
    ground_truth_path = os.path.join(dataset_dir, 'ground_truth.npy')
    if os.path.exists(ground_truth_path):
        dataset['ground_truth'] = np.load(ground_truth_path, allow_pickle=True).item()
    
    # Load metadata
    with open(os.path.join(dataset_dir, 'metadata.yaml'), 'r') as f:
        dataset['metadata'] = yaml.safe_load(f)
    
    return dataset

def load_model_predictions(predictions_dir):
    """
    Load model predictions from directory.
    
    Parameters:
    -----------
    predictions_dir : str
        Directory containing model predictions
        
    Returns:
    --------
    predictions : dict
        Dictionary containing prediction components
    """
    predictions = {}
    
    # Load detailed results if available
    detailed_results_path = os.path.join(predictions_dir, 'detailed_results.npz')
    if os.path.exists(detailed_results_path):
        detailed_results = np.load(detailed_results_path)
        predictions['predicted_expression'] = detailed_results['predicted_expression']
        predictions['true_expression'] = detailed_results['true_expression']
        predictions['cell_wise_spearman'] = detailed_results['cell_wise_spearman']
        predictions['gene_wise_spearman'] = detailed_results['gene_wise_spearman']
        predictions['test_indices'] = detailed_results['test_indices']
        predictions['gene_names'] = detailed_results['gene_names']
    
    # Load evaluation results if available
    evaluation_results_path = os.path.join(predictions_dir, 'evaluation_results.yaml')
    if os.path.exists(evaluation_results_path):
        with open(evaluation_results_path, 'r') as f:
            predictions['evaluation'] = yaml.safe_load(f)
    
    return predictions

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def visualize_dataset_overview(dataset, output_dir):
    """
    Generate overview visualizations of the synthetic dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dictionary containing dataset components
    output_dir : str
        Output directory for visualizations
    """
    print("Generating dataset overview visualizations...")
    
    gene_expression = dataset['gene_expression']
    cell_coordinates = dataset['cell_coordinates']
    region_labels = dataset['region_labels']
    quality_scores = dataset['quality_scores']
    gene_names = dataset['gene_names']
    
    # Create a figure with multiple subplots for overview
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # 1. Spatial distribution of cells colored by region
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(cell_coordinates[:, 0], cell_coordinates[:, 1], 
                         c=region_labels, cmap='tab10', s=10, alpha=0.7)
    ax1.set_title('Spatial Distribution by Region')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(scatter, cax=cax, label='Region')
    
    # 2. Spatial distribution of cells colored by quality score
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(cell_coordinates[:, 0], cell_coordinates[:, 1], 
                         c=quality_scores, cmap='viridis', s=10, alpha=0.7)
    ax2.set_title('Spatial Distribution by Quality Score')
    ax2.set_xlabel('X Coordinate')
    ax2.set_ylabel('Y Coordinate')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(scatter, cax=cax, label='Quality Score')
    
    # 3. PCA of gene expression colored by region
    ax3 = fig.add_subplot(gs[0, 2])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(gene_expression)
    scatter = ax3.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=region_labels, cmap='tab10', s=10, alpha=0.7)
    ax3.set_title('PCA of Gene Expression by Region')
    ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(scatter, cax=cax, label='Region')
    
    # 4. Gene expression distribution
    ax4 = fig.add_subplot(gs[1, 0])
    # Sample a few genes for visualization
    sample_genes = np.random.choice(gene_expression.shape[1], min(5, gene_expression.shape[1]), replace=False)
    for gene_idx in sample_genes:
        sns.kdeplot(gene_expression[:, gene_idx], ax=ax4, label=f'Gene {gene_idx}')
    ax4.set_title('Gene Expression Distributions')
    ax4.set_xlabel('Expression Level')
    ax4.set_ylabel('Density')
    ax4.legend()
    
    # 5. Gene-gene correlation heatmap
    ax5 = fig.add_subplot(gs[1, 1])
    # Sample a subset of genes for correlation heatmap
    n_sample_genes = min(20, gene_expression.shape[1])
    sample_genes_idx = np.random.choice(gene_expression.shape[1], n_sample_genes, replace=False)
    corr_matrix = np.corrcoef(gene_expression[:, sample_genes_idx].T)
    im = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax5.set_title('Gene-Gene Correlation Matrix')
    ax5.set_xticks([])
    ax5.set_yticks([])
    divider = make_axes_locatable(ax5)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Correlation')
    
    # 6. Dataset statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    metadata = dataset['metadata']
    stats_text = (
        f"Dataset Statistics:\n\n"
        f"Number of Cells: {metadata['n_cells']}\n"
        f"Number of Genes: {metadata['n_genes']}\n"
        f"Number of Regions: {metadata['n_regions']}\n"
        f"Number of Gene Modules: {metadata.get('n_gene_modules', 'N/A')}\n\n"
        f"Train/Val/Test Split:\n"
        f"Train: {metadata['n_train']} cells\n"
        f"Validation: {metadata['n_val']} cells\n"
        f"Test: {metadata['n_test']} cells\n\n"
        f"Synthetic Dataset: {metadata.get('synthetic', False)}"
    )
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
             fontsize=12, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dataset_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Dataset overview visualizations complete.")

def visualize_gene_modules(dataset, output_dir):
    """
    Generate visualizations of gene modules in the synthetic dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dictionary containing dataset components
    output_dir : str
        Output directory for visualizations
    """
    print("Generating gene module visualizations...")
    
    gene_expression = dataset['gene_expression']
    cell_coordinates = dataset['cell_coordinates']
    gene_names = dataset['gene_names']
    
    # Check if ground truth is available
    has_ground_truth = 'ground_truth' in dataset and 'gene_modules' in dataset['ground_truth']
    
    if has_ground_truth:
        gene_modules = dataset['ground_truth']['gene_modules']
        module_assignments = gene_modules['module_assignments']
        module_correlations = gene_modules['module_correlations']
        
        # Create a figure for each module
        for module_idx in range(len(module_correlations)):
            module_genes = np.where(module_assignments == module_idx)[0]
            
            if len(module_genes) <= 1:
                continue
            
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(2, 3, figure=fig)
            
            # 1. Module correlation matrix
            ax1 = fig.add_subplot(gs[0, 0])
            module_expr = gene_expression[:, module_genes]
            corr_matrix = np.corrcoef(module_expr.T)
            im = ax1.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax1.set_title(f'Module {module_idx} Correlation Matrix')
            ax1.set_xticks([])
            ax1.set_yticks([])
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax, label='Correlation')
            
            # 2. Spatial distribution of module expression
            ax2 = fig.add_subplot(gs[0, 1])
            module_mean_expr = np.mean(module_expr, axis=1)
            scatter = ax2.scatter(cell_coordinates[:, 0], cell_coordinates[:, 1], 
                                 c=module_mean_expr, cmap='viridis', s=10, alpha=0.7)
            ax2.set_title(f'Spatial Distribution of Module {module_idx} Expression')
            ax2.set_xlabel('X Coordinate')
            ax2.set_ylabel('Y Coordinate')
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(scatter, cax=cax, label='Mean Expression')
            
            # 3. Module expression distribution
            ax3 = fig.add_subplot(gs[0, 2])
            sns.kdeplot(module_mean_expr, ax=ax3, color='blue', fill=True)
            ax3.set_title(f'Module {module_idx} Expression Distribution')
            ax3.set_xlabel('Mean Expression Level')
            ax3.set_ylabel('Density')
            
            # 4. Heatmap of module genes across cells
            ax4 = fig.add_subplot(gs[1, :])
            # Sample cells for better visualization
            n_sample_cells = min(100, gene_expression.shape[0])
            sample_cells_idx = np.random.choice(gene_expression.shape[0], n_sample_cells, replace=False)
            
            # Scale data for better visualization
            module_expr_sample = module_expr[sample_cells_idx]
            module_expr_scaled = (module_expr_sample - np.min(module_expr_sample, axis=0)) / \
                               (np.max(module_expr_sample, axis=0) - np.min(module_expr_sample, axis=0) + 1e-10)
            
            im = ax4.imshow(module_expr_scaled.T, aspect='auto', cmap='viridis')
            ax4.set_title(f'Module {module_idx} Gene Expression Across Cells')
            ax4.set_xlabel('Cells (sampled)')
            ax4.set_ylabel('Genes in Module')
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes("right", size="2%", pad=0.1)
            plt.colorbar(im, cax=cax, label='Scaled Expression')
            
            # Add module statistics as text
            module_stats = (
                f"Module {module_idx} Statistics:\n"
                f"Number of Genes: {len(module_genes)}\n"
                f"Expected Correlation: {module_correlations[module_idx]:.4f}\n"
                f"Observed Correlation: {np.mean(corr_matrix[np.triu_indices(len(corr_matrix), k=1)]):.4f}"
            )
            ax4.text(0.01, -0.15, module_stats, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'module_{module_idx}_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
    else:
        # If no ground truth is available, use clustering to identify modules
        from scipy.cluster.hierarchy import linkage, fcluster
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(gene_expression.T)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(1 - np.abs(corr_matrix), method='ward')
        
        # Determine optimal number of clusters (simplified approach)
        n_clusters = min(10, gene_expression.shape[1] // 5)
        
        # Get cluster assignments
        cluster_assignments = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        # Create a figure for each detected module
        for cluster_idx in range(1, n_clusters + 1):
            cluster_genes = np.where(cluster_assignments == cluster_idx)[0]
            
            if len(cluster_genes) <= 1:
                continue
            
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(18, 12))
            gs = gridspec.GridSpec(2, 3, figure=fig)
            
            # 1. Module correlation matrix
            ax1 = fig.add_subplot(gs[0, 0])
            cluster_expr = gene_expression[:, cluster_genes]
            cluster_corr_matrix = np.corrcoef(cluster_expr.T)
            im = ax1.imshow(cluster_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax1.set_title(f'Detected Module {cluster_idx} Correlation Matrix')
            ax1.set_xticks([])
            ax1.set_yticks([])
            divider = make_axes_locatable(ax1)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax, label='Correlation')
            
            # 2. Spatial distribution of module expression
            ax2 = fig.add_subplot(gs[0, 1])
            cluster_mean_expr = np.mean(cluster_expr, axis=1)
            scatter = ax2.scatter(cell_coordinates[:, 0], cell_coordinates[:, 1], 
                                 c=cluster_mean_expr, cmap='viridis', s=10, alpha=0.7)
            ax2.set_title(f'Spatial Distribution of Detected Module {cluster_idx} Expression')
            ax2.set_xlabel('X Coordinate')
            ax2.set_ylabel('Y Coordinate')
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(scatter, cax=cax, label='Mean Expression')
            
            # 3. Module expression distribution
            ax3 = fig.add_subplot(gs[0, 2])
            sns.kdeplot(cluster_mean_expr, ax=ax3, color='blue', fill=True)
            ax3.set_title(f'Detected Module {cluster_idx} Expression Distribution')
            ax3.set_xlabel('Mean Expression Level')
            ax3.set_ylabel('Density')
            
            # 4. Heatmap of module genes across cells
            ax4 = fig.add_subplot(gs[1, :])
            # Sample cells for better visualization
            n_sample_cells = min(100, gene_expression.shape[0])
            sample_cells_idx = np.random.choice(gene_expression.shape[0], n_sample_cells, replace=False)
            
            # Scale data for better visualization
            cluster_expr_sample = cluster_expr[sample_cells_idx]
            cluster_expr_scaled = (cluster_expr_sample - np.min(cluster_expr_sample, axis=0)) / \
                                (np.max(cluster_expr_sample, axis=0) - np.min(cluster_expr_sample, axis=0) + 1e-10)
            
            im = ax4.imshow(cluster_expr_scaled.T, aspect='auto', cmap='viridis')
            ax4.set_title(f'Detected Module {cluster_idx} Gene Expression Across Cells')
            ax4.set_xlabel('Cells (sampled)')
            ax4.set_ylabel('Genes in Module')
            divider = make_axes_locatable(ax4)
            cax = divider.append_axes("right", size="2%", pad=0.1)
            plt.colorbar(im, cax=cax, label='Scaled Expression')
            
            # Add module statistics as text
            module_stats = (
                f"Detected Module {cluster_idx} Statistics:\n"
                f"Number of Genes: {len(cluster_genes)}\n"
                f"Average Correlation: {np.mean(cluster_corr_matrix[np.triu_indices(len(cluster_corr_matrix), k=1)]):.4f}"
            )
            ax4.text(0.01, -0.15, module_stats, transform=ax4.transAxes, 
                    fontsize=10, verticalalignment='top')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'detected_module_{cluster_idx}_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    print("Gene module visualizations complete.")

def visualize_region_properties(dataset, output_dir):
    """
    Generate visualizations of region properties in the synthetic dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dictionary containing dataset components
    output_dir : str
        Output directory for visualizations
    """
    print("Generating region property visualizations...")
    
    gene_expression = dataset['gene_expression']
    cell_coordinates = dataset['cell_coordinates']
    region_labels = dataset['region_labels']
    gene_names = dataset['gene_names']
    
    # Get unique regions
    unique_regions = np.unique(region_labels)
    n_regions = len(unique_regions)
    
    # Create a figure for region overview
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # 1. Spatial distribution of cells colored by region
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(cell_coordinates[:, 0], cell_coordinates[:, 1], 
                         c=region_labels, cmap='tab10', s=10, alpha=0.7)
    ax1.set_title('Spatial Distribution of Regions')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(scatter, cax=cax, label='Region')
    
    # 2. PCA of gene expression colored by region
    ax2 = fig.add_subplot(gs[0, 1])
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(gene_expression)
    scatter = ax2.scatter(pca_result[:, 0], pca_result[:, 1], 
                         c=region_labels, cmap='tab10', s=10, alpha=0.7)
    ax2.set_title('PCA of Gene Expression by Region')
    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(scatter, cax=cax, label='Region')
    
    # 3. UMAP of gene expression colored by region
    ax3 = fig.add_subplot(gs[0, 2])
    try:
        umap_model = UMAP(n_components=2, random_state=42)
        umap_result = umap_model.fit_transform(gene_expression)
        scatter = ax3.scatter(umap_result[:, 0], umap_result[:, 1], 
                             c=region_labels, cmap='tab10', s=10, alpha=0.7)
        ax3.set_title('UMAP of Gene Expression by Region')
        ax3.set_xlabel('UMAP1')
        ax3.set_ylabel('UMAP2')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(scatter, cax=cax, label='Region')
    except:
        ax3.text(0.5, 0.5, 'UMAP visualization failed', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('UMAP Visualization (Failed)')
    
    # 4. Calculate mean expression for each region
    region_means = np.zeros((n_regions, gene_expression.shape[1]))
    for i, region in enumerate(unique_regions):
        region_cells = region_labels == region
        region_means[i] = np.mean(gene_expression[region_cells], axis=0)
    
    # Calculate variance of means across regions to find differentially expressed genes
    mean_variance = np.var(region_means, axis=0)
    top_de_genes = np.argsort(mean_variance)[::-1][:20]
    
    # 4. Heatmap of top differentially expressed genes by region
    ax4 = fig.add_subplot(gs[1, :])
    
    # Prepare data for heatmap
    heatmap_data = np.zeros((len(top_de_genes), n_regions))
    
    for i, gene_idx in enumerate(top_de_genes):
        for j, region in enumerate(unique_regions):
            region_cells = region_labels == region
            heatmap_data[i, j] = np.mean(gene_expression[region_cells, gene_idx])
    
    # Scale data for better visualization
    heatmap_data_scaled = (heatmap_data - np.min(heatmap_data, axis=1, keepdims=True)) / \
                         (np.max(heatmap_data, axis=1, keepdims=True) - np.min(heatmap_data, axis=1, keepdims=True) + 1e-10)
    
    im = ax4.imshow(heatmap_data_scaled, cmap='viridis', aspect='auto')
    ax4.set_title('Top Differentially Expressed Genes by Region')
    ax4.set_xlabel('Region')
    ax4.set_ylabel('Gene')
    ax4.set_xticks(np.arange(n_regions))
    ax4.set_xticklabels([f'Region {r}' for r in unique_regions])
    ax4.set_yticks(np.arange(len(top_de_genes)))
    ax4.set_yticklabels([f'Gene {g}' for g in top_de_genes])
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    plt.colorbar(im, cax=cax, label='Scaled Expression')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'region_properties.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual visualizations for each region
    for region in unique_regions:
        region_cells = region_labels == region
        
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # 1. Spatial distribution of this region
        ax1 = fig.add_subplot(gs[0, 0])
        # Plot all cells in gray
        ax1.scatter(cell_coordinates[:, 0], cell_coordinates[:, 1], 
                   color='lightgray', s=10, alpha=0.3)
        # Highlight cells in this region
        ax1.scatter(cell_coordinates[region_cells, 0], cell_coordinates[region_cells, 1], 
                   color='red', s=10, alpha=0.7)
        ax1.set_title(f'Spatial Distribution of Region {region}')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        
        # 2. Gene expression distribution for this region vs. others
        ax2 = fig.add_subplot(gs[0, 1])
        # Calculate mean expression for this region
        region_mean_expr = np.mean(gene_expression[region_cells], axis=0)
        other_mean_expr = np.mean(gene_expression[~region_cells], axis=0)
        
        # Create scatter plot of mean expression: this region vs. others
        ax2.scatter(region_mean_expr, other_mean_expr, alpha=0.5)
        
        # Add diagonal line
        min_val = min(np.min(region_mean_expr), np.min(other_mean_expr))
        max_val = max(np.max(region_mean_expr), np.max(other_mean_expr))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax2.set_title(f'Region {region} vs. Other Regions Mean Expression')
        ax2.set_xlabel(f'Region {region} Mean Expression')
        ax2.set_ylabel('Other Regions Mean Expression')
        
        # 3. Top differentially expressed genes for this region
        ax3 = fig.add_subplot(gs[1, :])
        
        # Calculate fold change for each gene
        fold_change = region_mean_expr / (other_mean_expr + 1e-10)  # Add small constant to avoid division by zero
        log2_fold_change = np.log2(fold_change)
        
        # Get top genes by absolute log2 fold change
        top_genes_idx = np.argsort(np.abs(log2_fold_change))[::-1][:20]
        
        # Create bar plot
        bars = ax3.bar(np.arange(len(top_genes_idx)), log2_fold_change[top_genes_idx])
        
        # Color bars by direction of change
        for i, bar in enumerate(bars):
            if log2_fold_change[top_genes_idx[i]] > 0:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_title(f'Top Differentially Expressed Genes in Region {region}')
        ax3.set_xlabel('Gene')
        ax3.set_ylabel('Log2 Fold Change')
        ax3.set_xticks(np.arange(len(top_genes_idx)))
        ax3.set_xticklabels([f'Gene {g}' for g in top_genes_idx], rotation=90)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'region_{region}_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Region property visualizations complete.")

def visualize_model_performance(dataset, predictions, output_dir):
    """
    Generate visualizations of model performance on synthetic dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dictionary containing dataset components
    predictions : dict
        Dictionary containing model predictions
    output_dir : str
        Output directory for visualizations
    """
    print("Generating model performance visualizations...")
    
    # Check if predictions are available
    if not predictions or 'predicted_expression' not in predictions:
        print("No predictions available. Skipping model performance visualizations.")
        return
    
    gene_expression = dataset['gene_expression']
    cell_coordinates = dataset['cell_coordinates']
    region_labels = dataset['region_labels']
    gene_names = dataset['gene_names']
    test_indices = dataset['test_indices']
    
    predicted_expression = predictions['predicted_expression']
    true_expression = predictions['true_expression']
    cell_wise_spearman = predictions.get('cell_wise_spearman', None)
    gene_wise_spearman = predictions.get('gene_wise_spearman', None)
    
    # Create a figure for model performance overview
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(2, 3, figure=fig)
    
    # 1. Scatter plot of predicted vs. true expression (all genes, all cells)
    ax1 = fig.add_subplot(gs[0, 0])
    # Flatten arrays for overall comparison
    true_flat = true_expression.flatten()
    pred_flat = predicted_expression.flatten()
    
    # Create hexbin plot for density visualization
    hb = ax1.hexbin(true_flat, pred_flat, gridsize=50, cmap='viridis', 
                   bins='log', mincnt=1)
    
    # Add diagonal line
    min_val = min(np.min(true_flat), np.min(pred_flat))
    max_val = max(np.max(true_flat), np.max(pred_flat))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Calculate overall correlation
    overall_corr, _ = stats.spearmanr(true_flat, pred_flat)
    
    ax1.set_title(f'Predicted vs. True Expression\nOverall Spearman: {overall_corr:.4f}')
    ax1.set_xlabel('True Expression')
    ax1.set_ylabel('Predicted Expression')
    
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(hb, cax=cax, label='Count (log scale)')
    
    # 2. Cell-wise correlation distribution
    ax2 = fig.add_subplot(gs[0, 1])
    if cell_wise_spearman is not None:
        sns.histplot(cell_wise_spearman, kde=True, ax=ax2)
        ax2.axvline(x=np.mean(cell_wise_spearman), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(cell_wise_spearman):.4f}')
        ax2.set_title('Cell-wise Spearman Correlation Distribution')
        ax2.set_xlabel('Spearman Correlation')
        ax2.set_ylabel('Count')
        ax2.legend()
    else:
        # Calculate cell-wise correlation
        cell_corrs = []
        for i in range(true_expression.shape[0]):
            corr, _ = stats.spearmanr(true_expression[i], predicted_expression[i])
            cell_corrs.append(corr)
        
        sns.histplot(cell_corrs, kde=True, ax=ax2)
        ax2.axvline(x=np.mean(cell_corrs), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(cell_corrs):.4f}')
        ax2.set_title('Cell-wise Spearman Correlation Distribution')
        ax2.set_xlabel('Spearman Correlation')
        ax2.set_ylabel('Count')
        ax2.legend()
    
    # 3. Gene-wise correlation distribution
    ax3 = fig.add_subplot(gs[0, 2])
    if gene_wise_spearman is not None:
        sns.histplot(gene_wise_spearman, kde=True, ax=ax3)
        ax3.axvline(x=np.mean(gene_wise_spearman), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(gene_wise_spearman):.4f}')
        ax3.set_title('Gene-wise Spearman Correlation Distribution')
        ax3.set_xlabel('Spearman Correlation')
        ax3.set_ylabel('Count')
        ax3.legend()
    else:
        # Calculate gene-wise correlation
        gene_corrs = []
        for i in range(true_expression.shape[1]):
            corr, _ = stats.spearmanr(true_expression[:, i], predicted_expression[:, i])
            gene_corrs.append(corr)
        
        sns.histplot(gene_corrs, kde=True, ax=ax3)
        ax3.axvline(x=np.mean(gene_corrs), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(gene_corrs):.4f}')
        ax3.set_title('Gene-wise Spearman Correlation Distribution')
        ax3.set_xlabel('Spearman Correlation')
        ax3.set_ylabel('Count')
        ax3.legend()
    
    # 4. Spatial distribution of prediction quality
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Calculate cell-wise correlation if not provided
    if cell_wise_spearman is None:
        cell_wise_spearman = np.zeros(true_expression.shape[0])
        for i in range(true_expression.shape[0]):
            corr, _ = stats.spearmanr(true_expression[i], predicted_expression[i])
            cell_wise_spearman[i] = corr
    
    # Map test indices to original cell indices
    test_cell_coords = cell_coordinates[test_indices]
    
    scatter = ax4.scatter(test_cell_coords[:, 0], test_cell_coords[:, 1], 
                         c=cell_wise_spearman, cmap='RdYlGn', s=10, alpha=0.7,
                         vmin=0, vmax=1)
    ax4.set_title('Spatial Distribution of Prediction Quality')
    ax4.set_xlabel('X Coordinate')
    ax4.set_ylabel('Y Coordinate')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(scatter, cax=cax, label='Spearman Correlation')
    
    # 5. Prediction quality by region
    ax5 = fig.add_subplot(gs[1, 1])
    
    # Get region labels for test cells
    test_region_labels = region_labels[test_indices]
    
    # Calculate mean correlation by region
    unique_regions = np.unique(test_region_labels)
    region_corrs = []
    region_labels_list = []
    
    for region in unique_regions:
        region_mask = test_region_labels == region
        region_corr = np.mean(cell_wise_spearman[region_mask])
        region_corrs.append(region_corr)
        region_labels_list.append(f'Region {region}')
    
    # Create bar plot
    bars = ax5.bar(region_labels_list, region_corrs)
    
    # Color bars by correlation value
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.RdYlGn(region_corrs[i]))
    
    ax5.set_title('Prediction Quality by Region')
    ax5.set_xlabel('Region')
    ax5.set_ylabel('Mean Spearman Correlation')
    ax5.set_ylim(0, 1)
    
    # 6. Top and bottom predicted genes
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Calculate gene-wise correlation if not provided
    if gene_wise_spearman is None:
        gene_wise_spearman = np.zeros(true_expression.shape[1])
        for i in range(true_expression.shape[1]):
            corr, _ = stats.spearmanr(true_expression[:, i], predicted_expression[:, i])
            gene_wise_spearman[i] = corr
    
    # Get top and bottom genes
    n_top_bottom = 10
    top_genes_idx = np.argsort(gene_wise_spearman)[-n_top_bottom:][::-1]
    bottom_genes_idx = np.argsort(gene_wise_spearman)[:n_top_bottom]
    
    # Combine indices and create labels
    combined_idx = np.concatenate([top_genes_idx, bottom_genes_idx])
    combined_corrs = gene_wise_spearman[combined_idx]
    combined_labels = [f'Gene {i}' for i in combined_idx]
    
    # Create horizontal bar plot
    bars = ax6.barh(np.arange(len(combined_idx)), combined_corrs)
    
    # Color bars by correlation value
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.RdYlGn(max(0, min(1, (combined_corrs[i] + 1) / 2))))
    
    ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_title('Top and Bottom Predicted Genes')
    ax6.set_xlabel('Spearman Correlation')
    ax6.set_yticks(np.arange(len(combined_idx)))
    ax6.set_yticklabels(combined_labels)
    ax6.set_xlim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create detailed visualizations for a few genes
    n_genes_to_visualize = min(5, true_expression.shape[1])
    
    # Select a mix of well-predicted and poorly-predicted genes
    genes_to_visualize = []
    genes_to_visualize.extend(top_genes_idx[:n_genes_to_visualize // 2])
    genes_to_visualize.extend(bottom_genes_idx[:n_genes_to_visualize - len(genes_to_visualize)])
    
    for gene_idx in genes_to_visualize:
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig)
        
        # 1. Scatter plot of predicted vs. true expression for this gene
        ax1 = fig.add_subplot(gs[0, 0])
        true_gene = true_expression[:, gene_idx]
        pred_gene = predicted_expression[:, gene_idx]
        
        ax1.scatter(true_gene, pred_gene, alpha=0.5)
        
        # Add diagonal line
        min_val = min(np.min(true_gene), np.min(pred_gene))
        max_val = max(np.max(true_gene), np.max(pred_gene))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Calculate correlation
        gene_corr, _ = stats.spearmanr(true_gene, pred_gene)
        
        ax1.set_title(f'Gene {gene_idx} Predicted vs. True Expression\nSpearman: {gene_corr:.4f}')
        ax1.set_xlabel('True Expression')
        ax1.set_ylabel('Predicted Expression')
        
        # 2. Distribution of true and predicted expression
        ax2 = fig.add_subplot(gs[0, 1])
        sns.kdeplot(true_gene, ax=ax2, label='True', color='blue', fill=True, alpha=0.3)
        sns.kdeplot(pred_gene, ax=ax2, label='Predicted', color='red', fill=True, alpha=0.3)
        ax2.set_title(f'Gene {gene_idx} Expression Distribution')
        ax2.set_xlabel('Expression Level')
        ax2.set_ylabel('Density')
        ax2.legend()
        
        # 3. Spatial distribution of true expression
        ax3 = fig.add_subplot(gs[1, 0])
        scatter = ax3.scatter(test_cell_coords[:, 0], test_cell_coords[:, 1], 
                             c=true_gene, cmap='viridis', s=10, alpha=0.7)
        ax3.set_title(f'Gene {gene_idx} True Expression')
        ax3.set_xlabel('X Coordinate')
        ax3.set_ylabel('Y Coordinate')
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(scatter, cax=cax, label='Expression')
        
        # 4. Spatial distribution of predicted expression
        ax4 = fig.add_subplot(gs[1, 1])
        scatter = ax4.scatter(test_cell_coords[:, 0], test_cell_coords[:, 1], 
                             c=pred_gene, cmap='viridis', s=10, alpha=0.7)
        ax4.set_title(f'Gene {gene_idx} Predicted Expression')
        ax4.set_xlabel('X Coordinate')
        ax4.set_ylabel('Y Coordinate')
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(scatter, cax=cax, label='Expression')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'gene_{gene_idx}_prediction.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Model performance visualizations complete.")

def visualize_module_prediction(dataset, predictions, output_dir):
    """
    Generate visualizations of model performance on gene modules.
    
    Parameters:
    -----------
    dataset : dict
        Dictionary containing dataset components
    predictions : dict
        Dictionary containing model predictions
    output_dir : str
        Output directory for visualizations
    """
    print("Generating module prediction visualizations...")
    
    # Check if predictions and ground truth are available
    if not predictions or 'predicted_expression' not in predictions:
        print("No predictions available. Skipping module prediction visualizations.")
        return
    
    if 'ground_truth' not in dataset or 'gene_modules' not in dataset['ground_truth']:
        print("No ground truth gene modules available. Skipping module prediction visualizations.")
        return
    
    gene_expression = dataset['gene_expression']
    cell_coordinates = dataset['cell_coordinates']
    test_indices = dataset['test_indices']
    gene_modules = dataset['ground_truth']['gene_modules']
    module_assignments = gene_modules['module_assignments']
    module_correlations = gene_modules['module_correlations']
    
    predicted_expression = predictions['predicted_expression']
    true_expression = predictions['true_expression']
    
    # Create a figure for each module
    for module_idx in range(len(module_correlations)):
        module_genes = np.where(module_assignments == module_idx)[0]
        
        if len(module_genes) <= 1:
            continue
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(2, 3, figure=fig)
        
        # 1. Module correlation matrix - true expression
        ax1 = fig.add_subplot(gs[0, 0])
        true_module_expr = true_expression[:, module_genes]
        true_corr_matrix = np.corrcoef(true_module_expr.T)
        im = ax1.imshow(true_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax1.set_title(f'Module {module_idx} True Correlation Matrix')
        ax1.set_xticks([])
        ax1.set_yticks([])
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='Correlation')
        
        # 2. Module correlation matrix - predicted expression
        ax2 = fig.add_subplot(gs[0, 1])
        pred_module_expr = predicted_expression[:, module_genes]
        pred_corr_matrix = np.corrcoef(pred_module_expr.T)
        im = ax2.imshow(pred_corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_title(f'Module {module_idx} Predicted Correlation Matrix')
        ax2.set_xticks([])
        ax2.set_yticks([])
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='Correlation')
        
        # 3. Correlation difference matrix
        ax3 = fig.add_subplot(gs[0, 2])
        diff_matrix = pred_corr_matrix - true_corr_matrix
        im = ax3.imshow(diff_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax3.set_title(f'Module {module_idx} Correlation Difference\n(Predicted - True)')
        ax3.set_xticks([])
        ax3.set_yticks([])
        divider = make_axes_locatable(ax3)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax, label='Difference')
        
        # 4. Spatial distribution of module expression - true
        ax4 = fig.add_subplot(gs[1, 0])
        true_module_mean = np.mean(true_module_expr, axis=1)
        
        # Map test indices to original cell indices
        test_cell_coords = cell_coordinates[test_indices]
        
        scatter = ax4.scatter(test_cell_coords[:, 0], test_cell_coords[:, 1], 
                             c=true_module_mean, cmap='viridis', s=10, alpha=0.7)
        ax4.set_title(f'Module {module_idx} True Mean Expression')
        ax4.set_xlabel('X Coordinate')
        ax4.set_ylabel('Y Coordinate')
        divider = make_axes_locatable(ax4)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(scatter, cax=cax, label='Mean Expression')
        
        # 5. Spatial distribution of module expression - predicted
        ax5 = fig.add_subplot(gs[1, 1])
        pred_module_mean = np.mean(pred_module_expr, axis=1)
        
        scatter = ax5.scatter(test_cell_coords[:, 0], test_cell_coords[:, 1], 
                             c=pred_module_mean, cmap='viridis', s=10, alpha=0.7)
        ax5.set_title(f'Module {module_idx} Predicted Mean Expression')
        ax5.set_xlabel('X Coordinate')
        ax5.set_ylabel('Y Coordinate')
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(scatter, cax=cax, label='Mean Expression')
        
        # 6. Scatter plot of predicted vs. true module expression
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.scatter(true_module_mean, pred_module_mean, alpha=0.5)
        
        # Add diagonal line
        min_val = min(np.min(true_module_mean), np.min(pred_module_mean))
        max_val = max(np.max(true_module_mean), np.max(pred_module_mean))
        ax6.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Calculate correlation
        module_corr, _ = stats.spearmanr(true_module_mean, pred_module_mean)
        
        ax6.set_title(f'Module {module_idx} Mean Expression\nSpearman: {module_corr:.4f}')
        ax6.set_xlabel('True Mean Expression')
        ax6.set_ylabel('Predicted Mean Expression')
        
        # Add module statistics as text
        true_avg_corr = np.mean(true_corr_matrix[np.triu_indices(len(true_corr_matrix), k=1)])
        pred_avg_corr = np.mean(pred_corr_matrix[np.triu_indices(len(pred_corr_matrix), k=1)])
        
        module_stats = (
            f"Module {module_idx} Statistics:\n"
            f"Number of Genes: {len(module_genes)}\n"
            f"Expected Correlation: {module_correlations[module_idx]:.4f}\n"
            f"True Avg Correlation: {true_avg_corr:.4f}\n"
            f"Predicted Avg Correlation: {pred_avg_corr:.4f}\n"
            f"Mean Expression Correlation: {module_corr:.4f}"
        )
        
        # Add text to figure
        fig.text(0.5, 0.01, module_stats, ha='center', fontsize=12)
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for text
        plt.savefig(os.path.join(output_dir, f'module_{module_idx}_prediction.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Module prediction visualizations complete.")

def generate_visualizations(dataset_dir, predictions_dir=None, output_dir=None, analysis_type='all'):
    """
    Generate comprehensive visualizations for synthetic dataset analysis.
    
    Parameters:
    -----------
    dataset_dir : str
        Directory containing synthetic dataset
    predictions_dir : str
        Directory containing model predictions
    output_dir : str
        Output directory for visualizations
    analysis_type : str
        Type of analysis to perform ('all', 'dataset', 'model', 'comparison')
    """
    print(f"Generating visualizations for synthetic dataset in {dataset_dir}...")
    
    # Create output directory if needed
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, 'visualizations')
    
    ensure_dir(output_dir)
    
    # Load synthetic dataset
    dataset = load_synthetic_dataset(dataset_dir)
    
    # Load model predictions if available
    predictions = None
    if predictions_dir is not None and os.path.exists(predictions_dir):
        predictions = load_model_predictions(predictions_dir)
    
    # Generate visualizations based on analysis type
    if analysis_type in ['all', 'dataset']:
        # Create subdirectory for dataset visualizations
        dataset_vis_dir = os.path.join(output_dir, 'dataset')
        ensure_dir(dataset_vis_dir)
        
        # Generate dataset overview visualizations
        visualize_dataset_overview(dataset, dataset_vis_dir)
        
        # Generate gene module visualizations
        visualize_gene_modules(dataset, dataset_vis_dir)
        
        # Generate region property visualizations
        visualize_region_properties(dataset, dataset_vis_dir)
    
    if analysis_type in ['all', 'model'] and predictions is not None:
        # Create subdirectory for model visualizations
        model_vis_dir = os.path.join(output_dir, 'model')
        ensure_dir(model_vis_dir)
        
        # Generate model performance visualizations
        visualize_model_performance(dataset, predictions, model_vis_dir)
    
    if analysis_type in ['all', 'comparison'] and predictions is not None:
        # Create subdirectory for comparison visualizations
        comparison_vis_dir = os.path.join(output_dir, 'comparison')
        ensure_dir(comparison_vis_dir)
        
        # Generate module prediction visualizations
        visualize_module_prediction(dataset, predictions, comparison_vis_dir)
    
    print(f"Visualizations complete. Results saved to {output_dir}")

def main():
    """Main function."""
    args = parse_args()
    generate_visualizations(args.dataset_dir, args.predictions_dir, args.output_dir, args.analysis_type)

if __name__ == "__main__":
    main()

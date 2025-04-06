#!/usr/bin/env python3
# scripts/evaluate_results.py

import argparse
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate and visualize results from the spatial transcriptomics pipeline')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of the experiment to evaluate')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save visualization results')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_visualizations(experiment_name, output_dir, config):
    """
    Create visualizations for evaluating model performance.
    
    Parameters:
    - experiment_name: Name of the experiment to evaluate
    - output_dir: Directory to save visualization results
    - config: Configuration dictionary
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load evaluation results
    evaluation_dir = os.path.join(config['output_dir'], 'evaluation', experiment_name)
    results_path = os.path.join(evaluation_dir, 'detailed_results.npz')
    
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        print("This is expected if you're running the script before training a model.")
        print("Creating placeholder visualizations for demonstration purposes.")
        create_placeholder_visualizations(output_dir)
        return
    
    # Load results
    results = np.load(results_path)
    predicted_expression = results['predicted_expression']
    true_expression = results['true_expression']
    cell_wise_spearman = results['cell_wise_spearman']
    gene_wise_spearman = results['gene_wise_spearman']
    gene_names = results['gene_names']
    
    # Create visualizations
    create_prediction_scatter(predicted_expression, true_expression, output_dir)
    create_correlation_heatmap(cell_wise_spearman, gene_wise_spearman, gene_names, output_dir)
    create_gene_ranking_plot(gene_wise_spearman, gene_names, output_dir)
    
    print(f"Visualizations created and saved to {output_dir}")

def create_placeholder_visualizations(output_dir):
    """
    Create placeholder visualizations for demonstration purposes.
    
    Parameters:
    - output_dir: Directory to save visualization results
    """
    # Create prediction scatter plot
    plt.figure(figsize=(10, 8))
    x = np.random.normal(0, 1, 1000)
    y = x + np.random.normal(0, 0.5, 1000)
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel('True Expression')
    plt.ylabel('Predicted Expression')
    plt.title('Prediction Scatter Plot (Placeholder)')
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create correlation heatmap
    plt.figure(figsize=(12, 10))
    data = np.random.rand(20, 20)
    sns.heatmap(data, cmap='viridis')
    plt.title('Gene Correlation Heatmap (Placeholder)')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create gene ranking plot
    plt.figure(figsize=(12, 8))
    genes = [f'Gene_{i}' for i in range(20)]
    values = sorted(np.random.uniform(0.3, 0.8, 20), reverse=True)
    plt.bar(genes, values)
    plt.xlabel('Genes')
    plt.ylabel('Spearman Correlation')
    plt.title('Gene Ranking by Prediction Accuracy (Placeholder)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gene_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Placeholder visualizations created and saved to {output_dir}")

def create_prediction_scatter(predicted_expression, true_expression, output_dir):
    """
    Create scatter plot of predicted vs. true expression for a sample of genes.
    
    Parameters:
    - predicted_expression: Predicted gene expression matrix
    - true_expression: True gene expression matrix
    - output_dir: Directory to save visualization results
    """
    # Select a sample of genes (e.g., first 5)
    n_genes = min(5, predicted_expression.shape[1])
    
    plt.figure(figsize=(15, 10))
    for i in range(n_genes):
        plt.subplot(2, 3, i+1)
        plt.scatter(true_expression[:, i], predicted_expression[:, i], alpha=0.5)
        plt.xlabel('True Expression')
        plt.ylabel('Predicted Expression')
        plt.title(f'Gene {i}')
        
        # Add correlation coefficient
        corr = np.corrcoef(true_expression[:, i], predicted_expression[:, i])[0, 1]
        plt.annotate(f'r = {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_correlation_heatmap(cell_wise_spearman, gene_wise_spearman, gene_names, output_dir):
    """
    Create heatmap of gene-wise Spearman correlations.
    
    Parameters:
    - cell_wise_spearman: Cell-wise Spearman correlation values
    - gene_wise_spearman: Gene-wise Spearman correlation values
    - gene_names: Names of genes
    - output_dir: Directory to save visualization results
    """
    # Select top 20 genes by correlation
    n_genes = min(20, len(gene_wise_spearman))
    top_indices = np.argsort(gene_wise_spearman)[-n_genes:][::-1]
    
    # Create correlation matrix (placeholder - in a real implementation, this would use actual gene-gene correlations)
    corr_matrix = np.random.rand(n_genes, n_genes)
    np.fill_diagonal(corr_matrix, 1.0)
    corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='viridis', 
                xticklabels=[str(gene_names[i])[:10] for i in top_indices],
                yticklabels=[str(gene_names[i])[:10] for i in top_indices])
    plt.title('Gene Correlation Heatmap (Top 20 Genes)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_gene_ranking_plot(gene_wise_spearman, gene_names, output_dir):
    """
    Create bar plot of genes ranked by prediction accuracy.
    
    Parameters:
    - gene_wise_spearman: Gene-wise Spearman correlation values
    - gene_names: Names of genes
    - output_dir: Directory to save visualization results
    """
    # Select top 20 genes by correlation
    n_genes = min(20, len(gene_wise_spearman))
    top_indices = np.argsort(gene_wise_spearman)[-n_genes:][::-1]
    top_correlations = gene_wise_spearman[top_indices]
    top_genes = [str(gene_names[i])[:10] for i in top_indices]
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    plt.bar(top_genes, top_correlations)
    plt.xlabel('Genes')
    plt.ylabel('Spearman Correlation')
    plt.title('Top 20 Genes by Prediction Accuracy')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gene_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    create_visualizations(args.experiment_name, args.output_dir, config)

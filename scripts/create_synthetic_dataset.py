#!/usr/bin/env python3
"""
Create a synthetic dataset with well-defined statistical properties for model validation.
This script generates artificial gene expression data with known distributions, spatial patterns,
and gene-gene correlations to facilitate rigorous testing of model capacity and performance.
"""

import os
import argparse
import yaml
import numpy as np
import pandas as pd
import random
import shutil
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Create a synthetic dataset with known statistical properties')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for synthetic dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations of the synthetic data')
    parser.add_argument('--vis_dir', type=str, default=None, help='Directory for visualization outputs')
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_cell_coordinates(config, seed):
    """
    Generate cell coordinates with defined spatial patterns.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    cell_coordinates : numpy.ndarray
        Array of shape (n_cells, 2) containing x,y coordinates
    region_labels : numpy.ndarray
        Array of shape (n_cells,) containing region labels
    region_info : dict
        Dictionary containing region information (centers, radii, etc.)
    """
    np.random.seed(seed)
    
    n_cells = config.get('n_cells', 2000)
    n_regions = config.get('n_regions', 5)
    region_type = config.get('region_type', 'circular')
    space_size = config.get('space_size', 1000)
    
    # Initialize arrays
    cell_coordinates = np.zeros((n_cells, 2))
    region_labels = np.zeros(n_cells, dtype=int)
    
    if region_type == 'circular':
        # Generate circular regions
        region_info = generate_circular_regions(n_regions, space_size)
        
        # Assign cells to regions
        cells_per_region = n_cells // n_regions
        remaining_cells = n_cells % n_regions
        
        cell_idx = 0
        for region_idx, region in enumerate(region_info):
            # Calculate number of cells for this region
            n_region_cells = cells_per_region + (1 if region_idx < remaining_cells else 0)
            
            # Generate cells within the circular region
            for _ in range(n_region_cells):
                # Generate random angle and radius (with square root for uniform distribution)
                theta = np.random.uniform(0, 2 * np.pi)
                r = region['radius'] * np.sqrt(np.random.uniform(0, 1))
                
                # Convert to Cartesian coordinates
                x = region['center'][0] + r * np.cos(theta)
                y = region['center'][1] + r * np.sin(theta)
                
                # Store coordinates and region label
                cell_coordinates[cell_idx] = [x, y]
                region_labels[cell_idx] = region_idx
                cell_idx += 1
    
    elif region_type == 'voronoi':
        # Generate Voronoi regions
        region_info = generate_voronoi_regions(n_regions, space_size)
        
        # Generate random points within the space
        cell_coordinates = np.random.uniform(0, space_size, (n_cells, 2))
        
        # Assign region labels based on closest region center
        for i in range(n_cells):
            distances = [np.linalg.norm(cell_coordinates[i] - center) for center in region_info['centers']]
            region_labels[i] = np.argmin(distances)
    
    elif region_type == 'gradient':
        # Generate a continuous gradient across the space
        region_info = {'type': 'gradient'}
        
        # Generate random points within the space
        cell_coordinates = np.random.uniform(0, space_size, (n_cells, 2))
        
        # Assign region labels based on position along x-axis
        region_boundaries = np.linspace(0, space_size, n_regions + 1)
        for i in range(n_cells):
            for j in range(n_regions):
                if region_boundaries[j] <= cell_coordinates[i, 0] < region_boundaries[j + 1]:
                    region_labels[i] = j
                    break
    
    else:
        raise ValueError(f"Unknown region type: {region_type}")
    
    return cell_coordinates, region_labels, region_info

def generate_circular_regions(n_regions, space_size):
    """Generate circular regions with random centers and radii."""
    regions = []
    
    # Minimum distance between region centers
    min_distance = space_size / (n_regions * 0.8)
    
    # Minimum and maximum radius as a fraction of space_size
    min_radius = space_size * 0.05
    max_radius = space_size * 0.15
    
    # Generate region centers
    centers = []
    for _ in range(n_regions):
        while True:
            # Generate random center
            center = np.random.uniform(0.2 * space_size, 0.8 * space_size, 2)
            
            # Check if center is far enough from existing centers
            if not centers or all(np.linalg.norm(center - c) >= min_distance for c in centers):
                centers.append(center)
                break
    
    # Generate regions with random radii
    for center in centers:
        radius = np.random.uniform(min_radius, max_radius)
        regions.append({
            'center': center,
            'radius': radius
        })
    
    return regions

def generate_voronoi_regions(n_regions, space_size):
    """Generate Voronoi regions with random centers."""
    # Generate random centers
    centers = np.random.uniform(0.1 * space_size, 0.9 * space_size, (n_regions, 2))
    
    # Add points at the corners to bound the Voronoi diagram
    corners = np.array([
        [0, 0],
        [0, space_size],
        [space_size, 0],
        [space_size, space_size]
    ])
    
    # Combine centers and corners
    points = np.vstack([centers, corners])
    
    # Compute Voronoi diagram
    vor = Voronoi(points)
    
    return {
        'centers': centers,
        'voronoi': vor
    }

def generate_gene_modules(config, n_genes, seed):
    """
    Generate gene modules with correlated expression patterns.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
    n_genes : int
        Total number of genes
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    gene_modules : dict
        Dictionary containing gene module information
    """
    np.random.seed(seed)
    
    n_modules = config.get('n_gene_modules', 10)
    min_module_size = config.get('min_module_size', 5)
    max_module_size = config.get('max_module_size', 20)
    
    # Ensure we don't assign more genes than available
    total_module_capacity = n_modules * max_module_size
    if total_module_capacity < n_genes:
        max_module_size = n_genes // n_modules + 1
    
    # Initialize gene modules
    gene_modules = {
        'module_assignments': np.zeros(n_genes, dtype=int),  # Module assignment for each gene
        'module_sizes': [],                                  # Size of each module
        'module_correlations': []                            # Correlation strength within each module
    }
    
    # Assign genes to modules
    remaining_genes = list(range(n_genes))
    for module_idx in range(n_modules):
        # Determine module size
        if module_idx == n_modules - 1:
            # Last module gets all remaining genes
            module_size = len(remaining_genes)
        else:
            module_size = np.random.randint(min_module_size, min(max_module_size, len(remaining_genes) - min_module_size * (n_modules - module_idx - 1)))
        
        # Randomly select genes for this module
        module_genes = np.random.choice(remaining_genes, module_size, replace=False)
        
        # Remove selected genes from remaining genes
        remaining_genes = [g for g in remaining_genes if g not in module_genes]
        
        # Assign genes to this module
        gene_modules['module_assignments'][module_genes] = module_idx
        
        # Store module size
        gene_modules['module_sizes'].append(module_size)
        
        # Generate random correlation strength for this module
        correlation = np.random.uniform(0.5, 0.9)
        gene_modules['module_correlations'].append(correlation)
    
    # Generate gene names
    gene_names = np.array([f"gene_{i}" for i in range(n_genes)])
    
    # Add module-specific prefixes to gene names
    for i in range(n_genes):
        module_idx = gene_modules['module_assignments'][i]
        gene_names[i] = f"module{module_idx}_{gene_names[i]}"
    
    gene_modules['gene_names'] = gene_names
    
    return gene_modules

def generate_gene_expression(config, cell_coordinates, region_labels, gene_modules, seed):
    """
    Generate synthetic gene expression data with defined statistical properties.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
    cell_coordinates : numpy.ndarray
        Array of shape (n_cells, 2) containing x,y coordinates
    region_labels : numpy.ndarray
        Array of shape (n_cells,) containing region labels
    gene_modules : dict
        Dictionary containing gene module information
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    gene_expression : numpy.ndarray
        Array of shape (n_cells, n_genes) containing gene expression values
    """
    np.random.seed(seed)
    
    n_cells = len(cell_coordinates)
    n_genes = len(gene_modules['gene_names'])
    n_regions = len(np.unique(region_labels))
    
    # Initialize gene expression matrix
    gene_expression = np.zeros((n_cells, n_genes))
    
    # Get configuration parameters
    base_expression = config.get('base_expression', 5.0)
    expression_scale = config.get('expression_scale', 2.0)
    noise_level = config.get('noise_level', 0.2)
    spatial_effect_strength = config.get('spatial_effect_strength', 0.5)
    
    # Generate base expression profiles for each region
    region_profiles = np.zeros((n_regions, n_genes))
    for region in range(n_regions):
        # Each region has a unique expression profile
        region_profiles[region] = np.random.normal(base_expression, expression_scale, n_genes)
    
    # Generate module-specific expression patterns
    module_patterns = {}
    for module_idx in range(len(gene_modules['module_sizes'])):
        # Get genes in this module
        module_genes = np.where(gene_modules['module_assignments'] == module_idx)[0]
        
        # Generate correlated expression pattern for this module
        correlation = gene_modules['module_correlations'][module_idx]
        
        # Create correlation matrix
        corr_matrix = np.ones((len(module_genes), len(module_genes))) * correlation
        np.fill_diagonal(corr_matrix, 1.0)
        
        # Store module pattern
        module_patterns[module_idx] = {
            'genes': module_genes,
            'correlation': correlation,
            'corr_matrix': corr_matrix
        }
    
    # Generate expression values for each cell
    for cell_idx in range(n_cells):
        region = region_labels[cell_idx]
        
        # Start with the region-specific profile
        cell_expression = region_profiles[region].copy()
        
        # Add spatial effects
        x, y = cell_coordinates[cell_idx]
        spatial_effect = np.sin(x / 100) * np.cos(y / 100) * spatial_effect_strength
        cell_expression += spatial_effect
        
        # Add module-specific correlations
        for module_idx, pattern in module_patterns.items():
            module_genes = pattern['genes']
            correlation = pattern['correlation']
            
            # Generate correlated noise for this module
            # Higher correlation means more similar expression within the module
            module_noise = np.random.normal(0, noise_level, len(module_genes))
            
            # Apply module-specific noise
            cell_expression[module_genes] += module_noise
        
        # Ensure non-negative expression
        cell_expression = np.maximum(cell_expression, 0)
        
        # Store expression values
        gene_expression[cell_idx] = cell_expression
    
    # Apply negative binomial noise to mimic RNA-seq count data
    for gene_idx in range(n_genes):
        # Calculate mean and dispersion for negative binomial
        mean_expr = np.mean(gene_expression[:, gene_idx])
        dispersion = config.get('nb_dispersion', 0.1)
        
        # Convert to negative binomial parameters
        p = 1 / (1 + mean_expr * dispersion)
        n = mean_expr * p / (1 - p)
        
        # Generate negative binomial noise
        gene_expression[:, gene_idx] = stats.nbinom.rvs(n=n, p=p, size=n_cells)
    
    return gene_expression

def generate_quality_scores(config, cell_coordinates, gene_expression, region_labels, seed):
    """
    Generate synthetic quality scores for cells.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
    cell_coordinates : numpy.ndarray
        Array of shape (n_cells, 2) containing x,y coordinates
    gene_expression : numpy.ndarray
        Array of shape (n_cells, n_genes) containing gene expression values
    region_labels : numpy.ndarray
        Array of shape (n_cells,) containing region labels
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    quality_scores : numpy.ndarray
        Array of shape (n_cells,) containing quality scores
    """
    np.random.seed(seed)
    
    n_cells = len(cell_coordinates)
    
    # Initialize quality scores
    quality_scores = np.zeros(n_cells)
    
    # Base quality score is related to total gene expression
    total_expression = np.sum(gene_expression, axis=1)
    quality_scores = (total_expression - np.min(total_expression)) / (np.max(total_expression) - np.min(total_expression))
    
    # Add spatial component to quality
    # Cells near the center of their region have higher quality
    for cell_idx in range(n_cells):
        region = region_labels[cell_idx]
        
        # Find center of this region (mean of all cells in the region)
        region_cells = cell_coordinates[region_labels == region]
        region_center = np.mean(region_cells, axis=0)
        
        # Calculate distance to region center
        distance = np.linalg.norm(cell_coordinates[cell_idx] - region_center)
        
        # Normalize distance
        max_distance = np.max([np.linalg.norm(c - region_center) for c in region_cells])
        normalized_distance = distance / max_distance if max_distance > 0 else 0
        
        # Quality decreases with distance from center
        spatial_quality = 1 - normalized_distance
        
        # Combine with expression-based quality
        quality_scores[cell_idx] = 0.7 * quality_scores[cell_idx] + 0.3 * spatial_quality
    
    # Add random noise
    noise_level = config.get('quality_noise', 0.1)
    quality_scores += np.random.normal(0, noise_level, n_cells)
    
    # Ensure quality scores are between 0 and 1
    quality_scores = np.clip(quality_scores, 0, 1)
    
    return quality_scores

def create_synthetic_dataset(config, output_dir, seed=42, visualize=False, vis_dir=None):
    """
    Create a synthetic dataset with well-defined statistical properties.
    
    Parameters:
    -----------
    config : dict
        Configuration parameters
    output_dir : str
        Output directory for synthetic dataset
    seed : int
        Random seed for reproducibility
    visualize : bool
        Whether to generate visualizations
    vis_dir : str
        Directory for visualization outputs
        
    Returns:
    --------
    metadata : dict
        Metadata about the synthetic dataset
    """
    print("Creating synthetic dataset with known statistical properties...")
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Get configuration parameters
    n_cells = config.get('n_cells', 2000)
    n_genes = config.get('n_genes', 200)
    
    # Generate cell coordinates and region labels
    print("Generating cell coordinates and region labels...")
    cell_coordinates, region_labels, region_info = generate_cell_coordinates(config, seed)
    
    # Generate gene modules
    print("Generating gene modules...")
    gene_modules = generate_gene_modules(config, n_genes, seed)
    
    # Generate gene expression data
    print("Generating gene expression data...")
    gene_expression = generate_gene_expression(config, cell_coordinates, region_labels, gene_modules, seed)
    
    # Generate quality scores
    print("Generating quality scores...")
    quality_scores = generate_quality_scores(config, cell_coordinates, gene_expression, region_labels, seed)
    
    # Split into train, validation, and test sets
    print("Splitting into train, validation, and test sets...")
    train_indices, temp_indices = train_test_split(
        np.arange(n_cells), 
        test_size=config.get('val_test_size', 0.3),
        random_state=seed,
        stratify=region_labels
    )
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=seed,
        stratify=region_labels[temp_indices]
    )
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Save synthetic dataset
    print(f"Saving synthetic dataset to {output_dir}...")
    
    # Save gene expression data
    np.save(os.path.join(output_dir, 'gene_expression.npy'), gene_expression)
    
    # Save cell coordinates
    np.save(os.path.join(output_dir, 'cell_coordinates.npy'), cell_coordinates)
    
    # Save gene names
    np.save(os.path.join(output_dir, 'gene_names.npy'), gene_modules['gene_names'])
    
    # Save region labels
    np.save(os.path.join(output_dir, 'region_labels.npy'), region_labels)
    
    # Save indices
    np.save(os.path.join(output_dir, 'original_indices.npy'), np.arange(n_cells))
    np.save(os.path.join(output_dir, 'train_indices.npy'), train_indices)
    np.save(os.path.join(output_dir, 'val_indices.npy'), val_indices)
    np.save(os.path.join(output_dir, 'test_indices.npy'), test_indices)
    
    # Save quality scores
    np.save(os.path.join(output_dir, 'quality_scores.npy'), quality_scores)
    
    # Save ground truth information for validation
    ground_truth = {
        'gene_modules': gene_modules,
        'region_info': region_info
    }
    np.save(os.path.join(output_dir, 'ground_truth.npy'), ground_truth)
    
    # Create a metadata file
    metadata = {
        'n_cells': n_cells,
        'n_genes': n_genes,
        'n_train': len(train_indices),
        'n_val': len(val_indices),
        'n_test': len(test_indices),
        'n_regions': len(np.unique(region_labels)),
        'n_gene_modules': len(gene_modules['module_sizes']),
        'seed': seed,
        'synthetic': True
    }
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f)
    
    # Copy configuration file
    shutil.copy(config.get('config_path', 'config/synthetic_config.yaml'), os.path.join(output_dir, 'config.yaml'))
    
    # Generate visualizations if requested
    if visualize:
        if vis_dir is None:
            vis_dir = os.path.join(output_dir, 'visualizations')
        
        ensure_dir(vis_dir)
        generate_visualizations(cell_coordinates, gene_expression, region_labels, gene_modules, quality_scores, vis_dir)
    
    print("Synthetic dataset creation complete!")
    
    return metadata

def generate_visualizations(cell_coordinates, gene_expression, region_labels, gene_modules, quality_scores, vis_dir):
    """
    Generate visualizations of the synthetic dataset.
    
    Parameters:
    -----------
    cell_coordinates : numpy.ndarray
        Array of shape (n_cells, 2) containing x,y coordinates
    gene_expression : numpy.ndarray
        Array of shape (n_cells, n_genes) containing gene expression values
    region_labels : numpy.ndarray
        Array of shape (n_cells,) containing region labels
    gene_modules : dict
        Dictionary containing gene module information
    quality_scores : numpy.ndarray
        Array of shape (n_cells,) containing quality scores
    vis_dir : str
        Directory for visualization outputs
    """
    print("Generating visualizations...")
    
    # 1. Spatial plot of cells colored by region
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(cell_coordinates[:, 0], cell_coordinates[:, 1], c=region_labels, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, label='Region')
    plt.title('Spatial Distribution of Cells by Region')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(os.path.join(vis_dir, 'spatial_regions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Spatial plot of cells colored by quality score
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(cell_coordinates[:, 0], cell_coordinates[:, 1], c=quality_scores, cmap='viridis', s=10, alpha=0.7)
    plt.colorbar(scatter, label='Quality Score')
    plt.title('Spatial Distribution of Cells by Quality Score')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.savefig(os.path.join(vis_dir, 'spatial_quality.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Gene expression heatmap for top genes
    n_top_genes = min(50, gene_expression.shape[1])
    top_genes_idx = np.argsort(np.var(gene_expression, axis=0))[-n_top_genes:]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(gene_expression[:100, top_genes_idx], cmap='viridis', xticklabels=False, yticklabels=False)
    plt.title(f'Gene Expression Heatmap (Top {n_top_genes} Variable Genes, First 100 Cells)')
    plt.xlabel('Genes')
    plt.ylabel('Cells')
    plt.savefig(os.path.join(vis_dir, 'expression_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Gene-gene correlation matrix for a sample of genes
    n_sample_genes = min(30, gene_expression.shape[1])
    sample_genes_idx = np.random.choice(gene_expression.shape[1], n_sample_genes, replace=False)
    
    corr_matrix = np.corrcoef(gene_expression[:, sample_genes_idx].T)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
                xticklabels=False, yticklabels=False)
    plt.title('Gene-Gene Correlation Matrix (Sample of Genes)')
    plt.savefig(os.path.join(vis_dir, 'gene_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. PCA plot of cells colored by region
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(gene_expression)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=region_labels, cmap='tab10', s=10, alpha=0.7)
    plt.colorbar(scatter, label='Region')
    plt.title('PCA of Gene Expression Colored by Region')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.savefig(os.path.join(vis_dir, 'pca_regions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. UMAP plot of cells colored by region
    try:
        reducer = umap.UMAP()
        umap_result = reducer.fit_transform(gene_expression)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(umap_result[:, 0], umap_result[:, 1], c=region_labels, cmap='tab10', s=10, alpha=0.7)
        plt.colorbar(scatter, label='Region')
        plt.title('UMAP of Gene Expression Colored by Region')
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.savefig(os.path.join(vis_dir, 'umap_regions.png'), dpi=300, bbox_inches='tight')
        plt.close()
    except:
        print("Warning: UMAP visualization failed. Skipping.")
    
    # 7. Module-specific gene expression
    for module_idx in range(min(5, len(gene_modules['module_sizes']))):
        # Get genes in this module
        module_genes = np.where(gene_modules['module_assignments'] == module_idx)[0]
        
        if len(module_genes) == 0:
            continue
        
        # Calculate mean expression of module genes
        module_expression = np.mean(gene_expression[:, module_genes], axis=1)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(cell_coordinates[:, 0], cell_coordinates[:, 1], 
                             c=module_expression, cmap='viridis', s=10, alpha=0.7)
        plt.colorbar(scatter, label='Mean Expression')
        plt.title(f'Spatial Distribution of Module {module_idx} Expression')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.savefig(os.path.join(vis_dir, f'module_{module_idx}_expression.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Visualizations complete!")

def main():
    """Main function."""
    args = parse_args()
    config = load_config(args.config)
    
    # Store config path in config dict for later use
    config['config_path'] = args.config
    
    # Create synthetic dataset
    create_synthetic_dataset(config, args.output_dir, args.seed, args.visualize, args.vis_dir)

if __name__ == "__main__":
    main()

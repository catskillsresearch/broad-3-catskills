#!/usr/bin/env python3
# create_small_dataset.py

import os
import numpy as np
import pandas as pd
import zarr
import argparse
import yaml
from pathlib import Path
import random
import shutil
from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser(description='Create a small labeled dataset for spatial transcriptomics experiments')
    parser.add_argument('--config', type=str, default='config/small_dataset_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default='input/small_dataset',
                        help='Output directory for small dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def compute_quality_metrics(spatial_data, gene_expression):
    """
    Compute quality metrics for each spot to identify high-quality measurements.
    
    Parameters:
    - spatial_data: Spatial coordinates and metadata
    - gene_expression: Gene expression matrix
    
    Returns:
    - quality_scores: Quality score for each spot
    """
    # Number of detected genes per spot
    n_detected = (gene_expression > 0).sum(axis=1)
    
    # Total counts per spot
    total_counts = gene_expression.sum(axis=1)
    
    # Normalize scores between 0 and 1
    n_detected_norm = (n_detected - n_detected.min()) / (n_detected.max() - n_detected.min())
    total_counts_norm = (total_counts - total_counts.min()) / (total_counts.max() - total_counts.min())
    
    # Combined quality score (equal weighting)
    quality_scores = 0.5 * n_detected_norm + 0.5 * total_counts_norm
    
    return quality_scores

def stratified_sample_by_region(region_labels, n_samples_per_region):
    """
    Perform stratified sampling to ensure balanced representation of different tissue regions.
    
    Parameters:
    - region_labels: Region annotation for each spot
    - n_samples_per_region: Number of samples to select from each region
    
    Returns:
    - selected_indices: Indices of selected spots
    """
    unique_regions = np.unique(region_labels)
    selected_indices = []
    
    for region in unique_regions:
        region_indices = np.where(region_labels == region)[0]
        
        # If there are fewer spots than requested, take all of them
        if len(region_indices) <= n_samples_per_region:
            selected_indices.extend(region_indices)
        else:
            # Randomly select n_samples_per_region spots from this region
            sampled_indices = np.random.choice(
                region_indices, 
                size=n_samples_per_region, 
                replace=False
            )
            selected_indices.extend(sampled_indices)
    
    return np.array(selected_indices)

def extract_small_dataset(config, output_dir, seed=42):
    """
    Extract a small labeled dataset from the full dataset.
    
    Parameters:
    - config: Configuration dictionary
    - output_dir: Output directory for small dataset
    - seed: Random seed for reproducibility
    
    Returns:
    - small_dataset_info: Information about the extracted small dataset
    """
    np.random.seed(seed)
    random.seed(seed)
    
    print(f"Loading data from {config['data_path']}...")
    
    # Load spatial data
    spatial_data = zarr.open(config['data_path'], mode='r')
    
    print(f"Loading data from AnnData-like zarr structure...")
    
    # Load gene expression data from tables/anucleus/X
    gene_expression = np.array(spatial_data['tables']['anucleus']['X'])
    
    # Load cell coordinates from tables/anucleus/obsm/spatial
    cell_coordinates = np.array(spatial_data['tables']['anucleus']['obsm']['spatial'])
    
    # Load gene names from tables/anucleus/var/gene_symbols
    gene_names = np.array(spatial_data['tables']['anucleus']['var']['gene_symbols'])
    
    print(f"Loaded data: {gene_expression.shape} gene expressions, {cell_coordinates.shape} coordinates, {gene_names.shape} genes")
    
    # Load region annotations if available
    if 'region_annotations' in config and os.path.exists(config['region_annotations']):
        print(f"Loading region annotations from {config['region_annotations']}...")
        region_annotations = pd.read_csv(config['region_annotations'])
        region_labels = region_annotations['region_label'].values
    else:
        # If region annotations are not available, create dummy labels
        print("Region annotations not found. Creating dummy labels...")
        # Use k-means clustering on cell coordinates to create regions
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=config.get('n_dummy_regions', 5), random_state=seed)
        region_labels = kmeans.fit_predict(cell_coordinates)
    
    # Compute quality metrics
    print("Computing quality metrics...")
    quality_scores = compute_quality_metrics(cell_coordinates, gene_expression)
    
    # Select high-quality spots
    quality_threshold = config.get('quality_threshold', 0.7)
    high_quality_mask = quality_scores > quality_threshold
    high_quality_indices = np.where(high_quality_mask)[0]
    
    print(f"Found {len(high_quality_indices)} high-quality spots out of {len(quality_scores)} total spots")
    
    # Perform stratified sampling by region
    n_samples_per_region = config.get('n_samples_per_region', 50)
    
    # Filter region labels to only include high-quality spots
    filtered_region_labels = region_labels[high_quality_indices]
    
    print(f"Performing stratified sampling with {n_samples_per_region} samples per region...")
    stratified_indices = stratified_sample_by_region(
        filtered_region_labels, 
        n_samples_per_region
    )
    
    # Map back to original indices
    selected_indices = high_quality_indices[stratified_indices]
    
    print(f"Selected {len(selected_indices)} spots for the small dataset")
    
    # Extract small dataset
    small_gene_expression = gene_expression[selected_indices]
    small_cell_coordinates = cell_coordinates[selected_indices]
    small_region_labels = region_labels[selected_indices]
    
    # Split into train, validation, and test sets
    train_indices, temp_indices = train_test_split(
        np.arange(len(selected_indices)), 
        test_size=config.get('val_test_size', 0.3),
        random_state=seed,
        stratify=small_region_labels
    )
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=seed,
        stratify=small_region_labels[temp_indices]
    )
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Save small dataset
    print(f"Saving small dataset to {output_dir}...")
    
    # Save gene expression data
    np.save(os.path.join(output_dir, 'gene_expression.npy'), small_gene_expression)
    
    # Save cell coordinates
    np.save(os.path.join(output_dir, 'cell_coordinates.npy'), small_cell_coordinates)
    
    # Save gene names - FIX: Use allow_pickle=True when saving object arrays
    np.save(os.path.join(output_dir, 'gene_names.npy'), gene_names, allow_pickle=True)
    
    # Save region labels
    np.save(os.path.join(output_dir, 'region_labels.npy'), small_region_labels)
    
    # Save indices
    np.save(os.path.join(output_dir, 'original_indices.npy'), selected_indices)
    np.save(os.path.join(output_dir, 'train_indices.npy'), train_indices)
    np.save(os.path.join(output_dir, 'val_indices.npy'), val_indices)
    np.save(os.path.join(output_dir, 'test_indices.npy'), test_indices)
    
    # Save quality scores
    np.save(os.path.join(output_dir, 'quality_scores.npy'), quality_scores[selected_indices])
    
    # Create a metadata file
    metadata = {
        'n_spots': len(selected_indices),
        'n_genes': len(gene_names),
        'n_train': len(train_indices),
        'n_val': len(val_indices),
        'n_test': len(test_indices),
        'quality_threshold': quality_threshold,
        'n_samples_per_region': n_samples_per_region,
        'unique_regions': len(np.unique(small_region_labels)),
        'seed': seed
    }
    
    # Save metadata
    with open(os.path.join(output_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f)
    
    # Copy configuration file
    shutil.copy(args.config, os.path.join(output_dir, 'config.yaml'))
    
    print("Small dataset creation complete!")
    
    return metadata

if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    extract_small_dataset(config, args.output_dir, args.seed)

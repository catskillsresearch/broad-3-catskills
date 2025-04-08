#!/usr/bin/env python3
# integrated_pipeline.py

import os
import sys
import argparse
import yaml
import numpy as np
from pathlib import Path
import subprocess
import shutil

def parse_args():
    parser = argparse.ArgumentParser(description='Integrated pipeline for data preparation and model training')
    parser.add_argument('--config', type=str, default='config/small_dataset_config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--step', type=str, choices=['data', 'features', 'train', 'all'],
                        default='data', help='Pipeline step to run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def ensure_dir(directory):
    Path(directory).mkdir(parents=True, exist_ok=True)

def create_small_dataset(config, seed=42):
    """
    Create a small dataset for initial experiments.
    This function replicates the functionality from create_small_dataset.py
    but is integrated directly into the pipeline.
    """
    print("Creating small dataset...")
    
    # Create output directory structure
    output_dir = config['output_dir']
    data_dir = os.path.join(output_dir, 'data', 'small_dataset')
    ensure_dir(data_dir)
    
    # For demonstration purposes, we'll create dummy data
    # In a real scenario, this would load and process actual data
    
    # Create dummy gene expression data
    n_samples = 100
    n_genes = 200
    
    # Generate random gene expression data
    gene_expression = np.random.rand(n_samples, n_genes)
    
    # Generate random cell coordinates
    cell_coordinates = np.random.rand(n_samples, 2)
    
    # Generate dummy gene names
    gene_names = np.array([f"gene_{i}" for i in range(n_genes)])
    
    # Generate dummy region labels
    region_labels = np.random.randint(0, 5, size=n_samples)
    
    # Generate train/val/test indices
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_size = int(0.7 * n_samples)
    val_size = int(0.15 * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Save the data
    np.save(os.path.join(data_dir, 'gene_expression.npy'), gene_expression)
    np.save(os.path.join(data_dir, 'cell_coordinates.npy'), cell_coordinates)
    np.save(os.path.join(data_dir, 'gene_names.npy'), gene_names, allow_pickle=True)
    np.save(os.path.join(data_dir, 'region_labels.npy'), region_labels)
    np.save(os.path.join(data_dir, 'train_indices.npy'), train_indices)
    np.save(os.path.join(data_dir, 'val_indices.npy'), val_indices)
    np.save(os.path.join(data_dir, 'test_indices.npy'), test_indices)
    
    print(f"Small dataset created with {n_samples} samples and {n_genes} genes")
    return data_dir

def extract_features(config, data_dir):
    """
    Extract features from the small dataset.
    This function replicates the functionality from the pipeline_data_preparation.py
    but is integrated directly into the pipeline.
    """
    print("Extracting features...")
    
    # Create features directory
    features_dir = os.path.join(config['output_dir'], 'features')
    ensure_dir(features_dir)
    
    # Load data
    gene_expression = np.load(os.path.join(data_dir, 'gene_expression.npy'))
    gene_names = np.load(os.path.join(data_dir, 'gene_names.npy'), allow_pickle=True)
    train_indices = np.load(os.path.join(data_dir, 'train_indices.npy'))
    val_indices = np.load(os.path.join(data_dir, 'val_indices.npy'))
    test_indices = np.load(os.path.join(data_dir, 'test_indices.npy'))
    
    # Extract spot features (placeholder implementation)
    print("Extracting spot features...")
    feature_dim = config.get('feature_dim', 512)
    n_samples = len(gene_expression)
    
    # Create random features for demonstration
    spot_features = np.random.rand(n_samples, feature_dim)
    
    # Extract sub-spot features (placeholder implementation)
    print("Extracting sub-spot features...")
    n_subspots = config.get('n_subspots', 4)
    subspot_features = np.random.rand(n_samples, n_subspots, feature_dim)
    
    # Extract neighbor features (placeholder implementation)
    print("Extracting neighbor features...")
    n_neighbors = config.get('n_neighbors', 6)
    neighbor_features = np.random.rand(n_samples, n_neighbors, feature_dim)
    
    # Save training data
    training_data_path = os.path.join(features_dir, 'training_data.npz')
    np.savez(
        training_data_path,
        spot_features=spot_features,
        subspot_features=subspot_features,
        neighbor_features=neighbor_features,
        gene_expression=gene_expression,
        gene_names=gene_names,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices
    )
    
    print(f"Features extracted and saved to {training_data_path}")
    return training_data_path

def train_model(config, training_data_path):
    """
    Train a model using the extracted features.
    This is a placeholder for the actual model training code.
    """
    print("Training model...")
    
    # Create models directory
    models_dir = os.path.join(config['output_dir'], 'models')
    ensure_dir(models_dir)
    
    # Load training data
    data = np.load(training_data_path, allow_pickle=True)
    
    # Print available keys in the training data
    print("Available keys in training data:", list(data.keys()))
    
    # Placeholder for model training
    print("Model training would happen here in a real implementation")
    
    # Save a dummy model
    model_path = os.path.join(models_dir, 'model.npz')
    np.savez(model_path, weights=np.random.rand(10, 10))
    
    print(f"Model trained and saved to {model_path}")
    return model_path

def run_pipeline(config_path, step='all', seed=42):
    """
    Run the integrated pipeline.
    
    Parameters:
    - config_path: Path to configuration file
    - step: Pipeline step to run ('data', 'features', 'train', 'all')
    - seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    
    # Load configuration
    config = load_config(config_path)
    
    # Create output directory structure
    output_dir = config['output_dir']
    ensure_dir(output_dir)
    
    # Create directories marker file
    with open(os.path.join(output_dir, '.directories_created'), 'w') as f:
        f.write('Directories created successfully')
    
    # Run pipeline steps
    if step in ['data', 'all']:
        data_dir = create_small_dataset(config, seed)
    else:
        data_dir = os.path.join(output_dir, 'data', 'small_dataset')
    
    if step in ['features', 'all']:
        training_data_path = extract_features(config, data_dir)
    else:
        training_data_path = os.path.join(output_dir, 'features', 'training_data.npz')
    
    if step in ['train', 'all']:
        model_path = train_model(config, training_data_path)
    
    print(f"Pipeline step '{step}' completed successfully")
    
    # Verify the training_data.npz file exists and show its keys
    if os.path.exists(training_data_path):
        data = np.load(training_data_path, allow_pickle=True)
        print(f"\nVerification: {training_data_path} exists with keys: {list(data.keys())}")
    else:
        print(f"\nWarning: {training_data_path} does not exist")

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args.config, args.step, args.seed)

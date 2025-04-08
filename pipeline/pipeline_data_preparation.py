#!/usr/bin/env python3
# pipeline/tasks/data_preparation.py

import luigi
import os
import yaml
import subprocess
import numpy as np
from pathlib import Path

class EnsureDirectories(luigi.Task):
    """Ensure all necessary directories exist."""
    config_path = luigi.Parameter()
    
    def output(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Create a marker file to indicate completion
        return luigi.LocalTarget(os.path.join(config['output_dir'], '.directories_created'))
    
    def run(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get the output directory and make it absolute if it's not already
        output_dir = config['output_dir']
        if not os.path.isabs(output_dir):
            # Get the directory where the script is running from
            current_dir = os.path.abspath(os.getcwd())
            output_dir = os.path.join(current_dir, output_dir)
            print(f"Using absolute output directory path: {output_dir}")
        
        # Create all necessary directories with absolute paths
        directories = [
            output_dir,
            os.path.join(output_dir, 'data'),
            os.path.join(output_dir, 'features'),
            os.path.join(output_dir, 'models'),
            os.path.join(output_dir, 'predictions'),
            os.path.join(output_dir, 'evaluation')
        ]
        
        for directory in directories:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")
                raise
        
        # Create marker file
        with self.output().open('w') as f:
            f.write('Directories created successfully')


class CreateSmallDataset(luigi.Task):
    """Create a small labeled dataset for initial experiments."""
    config_path = luigi.Parameter()
    
    def requires(self):
        return EnsureDirectories(config_path=self.config_path)
    
    def output(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        small_dataset_dir = os.path.join(config['output_dir'], 'data', 'small_dataset')
        return luigi.LocalTarget(os.path.join(small_dataset_dir, 'metadata.yaml'))
    
    def run(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        small_dataset_dir = os.path.join(config['output_dir'], 'data', 'small_dataset')
        
        # Run the create_small_dataset.py script
        cmd = [
            'python', 'scripts/create_small_dataset.py',
            '--config', self.config_path,
            '--output_dir', small_dataset_dir,
            '--seed', str(config.get('seed', 42))
        ]
        
        subprocess.run(cmd, check=True)


class PrepareImagePatches(luigi.Task):
    """Extract image patches centered on each cell."""
    config_path = luigi.Parameter()
    
    def requires(self):
        return CreateSmallDataset(config_path=self.config_path)
    
    def output(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        features_dir = os.path.join(config['output_dir'], 'features')
        return luigi.LocalTarget(os.path.join(features_dir, 'image_patches.npz'))
    
    def run(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        small_dataset_dir = os.path.join(config['output_dir'], 'data', 'small_dataset')
        features_dir = os.path.join(config['output_dir'], 'features')
        
        # Load cell coordinates
        cell_coordinates = np.load(os.path.join(small_dataset_dir, 'cell_coordinates.npy'))
        
        # Load original indices
        original_indices = np.load(os.path.join(small_dataset_dir, 'original_indices.npy'))
        
        # Load train/val/test indices
        train_indices = np.load(os.path.join(small_dataset_dir, 'train_indices.npy'))
        val_indices = np.load(os.path.join(small_dataset_dir, 'val_indices.npy'))
        test_indices = np.load(os.path.join(small_dataset_dir, 'test_indices.npy'))
        
        # Extract image patches (placeholder implementation)
        # In a real implementation, this would load the H&E images and extract patches
        print("Extracting image patches...")
        
        # Placeholder: Create random patches for demonstration
        n_samples = len(cell_coordinates)
        patch_size = config.get('patch_size', 64)
        n_channels = 3  # RGB
        
        # Create random patches (this would be replaced with actual patch extraction)
        patches = np.random.rand(n_samples, patch_size, patch_size, n_channels)
        
        # Save patches
        np.savez(
            self.output().path,
            patches=patches,
            cell_coordinates=cell_coordinates,
            original_indices=original_indices,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices
        )
        
        print(f"Saved {n_samples} image patches to {self.output().path}")


class ExtractMultiLevelFeatures(luigi.Task):
    """Extract multi-level features (spot, sub-spot, neighbor) using DeepSpot approach."""
    config_path = luigi.Parameter()
    
    def requires(self):
        return PrepareImagePatches(config_path=self.config_path)
    
    def output(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        features_dir = os.path.join(config['output_dir'], 'features')
        return luigi.LocalTarget(os.path.join(features_dir, 'multilevel_features.npz'))
    
    def run(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        features_dir = os.path.join(config['output_dir'], 'features')
        
        # Load image patches
        data = np.load(self.requires().output().path)
        patches = data['patches']
        cell_coordinates = data['cell_coordinates']
        train_indices = data['train_indices']
        val_indices = data['val_indices']
        test_indices = data['test_indices']
        
        # Extract spot features (placeholder implementation)
        # In a real implementation, this would use a pre-trained model like ResNet50
        print("Extracting spot features...")
        feature_dim = config.get('feature_dim', 512)
        n_samples = len(patches)
        
        # Placeholder: Create random features for demonstration
        spot_features = np.random.rand(n_samples, feature_dim)
        
        # Extract sub-spot features (placeholder implementation)
        print("Extracting sub-spot features...")
        n_subspots = config.get('n_subspots', 4)
        subspot_features = np.random.rand(n_samples, n_subspots, feature_dim)
        
        # Extract neighbor features (placeholder implementation)
        print("Extracting neighbor features...")
        n_neighbors = config.get('n_neighbors', 6)
        neighbor_features = np.random.rand(n_samples, n_neighbors, feature_dim)
        
        # Save features
        np.savez(
            self.output().path,
            spot_features=spot_features,
            subspot_features=subspot_features,
            neighbor_features=neighbor_features,
            cell_coordinates=cell_coordinates,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices
        )
        
        print(f"Saved multi-level features to {self.output().path}")


class PrepareTrainingData(luigi.Task):
    """Prepare training data with features and gene expression."""
    config_path = luigi.Parameter()
    
    def requires(self):
        return {
            'features': ExtractMultiLevelFeatures(config_path=self.config_path),
            'dataset': CreateSmallDataset(config_path=self.config_path)
        }
    
    def output(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        features_dir = os.path.join(config['output_dir'], 'features')
        return luigi.LocalTarget(os.path.join(features_dir, 'training_data.npz'))
    
    def run(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        small_dataset_dir = os.path.join(config['output_dir'], 'data', 'small_dataset')
        
        # Load features
        features_data = np.load(self.requires()['features'].output().path)
        spot_features = features_data['spot_features']
        subspot_features = features_data['subspot_features']
        neighbor_features = features_data['neighbor_features']
        train_indices = features_data['train_indices']
        val_indices = features_data['val_indices']
        test_indices = features_data['test_indices']
        
        # Load gene expression - FIX: Added allow_pickle=True
        gene_expression = np.load(os.path.join(small_dataset_dir, 'gene_expression.npy'))
        gene_names = np.load(os.path.join(small_dataset_dir, 'gene_names.npy'), allow_pickle=True)
        
        # Save training data
        # Create X and y data for hyperparameter search
        # X should be the concatenated features
        n_samples = len(spot_features)
        feature_dim = spot_features.shape[1]
        n_subspots = subspot_features.shape[1]
        n_neighbors = neighbor_features.shape[1]
        
        X = np.concatenate([
            spot_features,
            subspot_features.reshape(n_samples, n_subspots * feature_dim),
            neighbor_features.reshape(n_samples, n_neighbors * feature_dim)
        ], axis=1)
        
        # y should be the gene expression
        y = gene_expression
        
        np.savez(
            self.output().path,
            # Original keys
            spot_features=spot_features,
            subspot_features=subspot_features,
            neighbor_features=neighbor_features,
            gene_expression=gene_expression,
            gene_names=gene_names,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            # Keys needed by hyperparameter search
            X=X,
            y=y
        )
        
        print(f"Prepared training data with {len(gene_names)} genes and saved to {self.output().path}")

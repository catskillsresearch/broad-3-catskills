#!/usr/bin/env python3
# pipeline/pipeline_data_preparation.py

import luigi
import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

class EnsureDirectories(luigi.Task):
    """Ensure all required directories exist for the pipeline."""
    config_path = luigi.Parameter(description="Path to the configuration file")
    
    def output(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get output directory from config
        output_dir = config.get('output_dir', 'output')
        
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        # Create a marker file to indicate directories are created
        return luigi.LocalTarget(os.path.join(output_dir, 'directories_created.txt'))
    
    def run(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get output directory from config
        output_dir = config.get('output_dir', 'output')
        
        # Convert to absolute path if it's a relative path
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        # Define all required directories
        required_dirs = [
            output_dir,
            os.path.join(output_dir, 'features'),
            os.path.join(output_dir, 'data'),
            os.path.join(output_dir, 'models'),
            os.path.join(output_dir, 'predictions'),
            os.path.join(output_dir, 'evaluation'),
            os.path.join(output_dir, 'visualizations'),
            os.path.join(output_dir, 'small_dataset'),
            os.path.join(output_dir, 'small_dataset', 'features')
        ]
        
        # Create all directories
        for directory in required_dirs:
            os.makedirs(directory, exist_ok=True)
            print(f"Created directory: {directory}")
        
        # Create a marker file to indicate directories are created
        with open(self.output().path, 'w') as f:
            f.write(f"Directories created at {os.path.basename(self.config_path)}")
        
        print(f"All required directories created successfully")


class PrepareImagePatches(luigi.Task):
    """Prepare image patches for feature extraction."""
    config_path = luigi.Parameter(description="Path to the configuration file")
    
    def requires(self):
        return EnsureDirectories(config_path=self.config_path)
    
    def output(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        return luigi.LocalTarget(os.path.join(output_dir, 'features', 'image_patches.npz'))
    
    def run(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        # Placeholder: Create dummy image patches for demonstration
        n_cells = 100
        patch_size = 64
        
        # Create random image patches
        image_patches = np.random.rand(n_cells, patch_size, patch_size, 3)
        
        # Save image patches
        np.savez(self.output().path, image_patches=image_patches)
        
        print(f"Image patches prepared and saved to {self.output().path}")


class ExtractFeatures(luigi.Task):
    """Extract features from image patches."""
    config_path = luigi.Parameter(description="Path to the configuration file")
    
    def requires(self):
        return PrepareImagePatches(config_path=self.config_path)
    
    def output(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        return luigi.LocalTarget(os.path.join(output_dir, 'features', 'extracted_features.npz'))
    
    def run(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        # Load image patches
        data = np.load(self.requires().output().path)
        image_patches = data['image_patches']
        
        # Placeholder: Extract features from image patches
        n_cells = image_patches.shape[0]
        feature_dim = 128
        
        # Create random features
        spot_features = np.random.rand(n_cells, feature_dim)
        subspot_features = np.random.rand(n_cells, feature_dim // 2)
        neighbor_features = np.random.rand(n_cells, feature_dim // 4)
        
        # Save extracted features
        np.savez(
            self.output().path,
            spot_features=spot_features,
            subspot_features=subspot_features,
            neighbor_features=neighbor_features
        )
        
        print(f"Features extracted and saved to {self.output().path}")


class PrepareTrainingData(luigi.Task):
    """Prepare training data for model training."""
    config_path = luigi.Parameter(description="Path to the configuration file")
    
    def requires(self):
        return ExtractFeatures(config_path=self.config_path)
    
    def output(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        return luigi.LocalTarget(os.path.join(output_dir, 'data', 'training_data.npz'))
    
    def run(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        # Load extracted features
        data = np.load(self.requires().output().path)
        spot_features = data['spot_features']
        subspot_features = data['subspot_features']
        neighbor_features = data['neighbor_features']
        
        # Placeholder: Create dummy gene expression data
        n_cells = spot_features.shape[0]
        n_genes = 1000
        
        # Create random gene expression data
        gene_expression = np.random.rand(n_cells, n_genes)
        
        # Create random cell coordinates
        cell_coordinates = np.random.rand(n_cells, 2) * 100
        
        # Create random region labels
        region_labels = np.random.randint(0, 3, n_cells)
        
        # Create random quality scores
        quality_scores = np.random.rand(n_cells)
        
        # Create random gene names
        gene_names = [f'Gene_{i}' for i in range(n_genes)]
        
        # Split data into train, validation, and test sets
        indices = np.arange(n_cells)
        np.random.shuffle(indices)
        
        train_size = int(0.7 * n_cells)
        val_size = int(0.15 * n_cells)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Save training data
        np.savez(
            self.output().path,
            spot_features=spot_features,
            subspot_features=subspot_features,
            neighbor_features=neighbor_features,
            gene_expression=gene_expression,
            cell_coordinates=cell_coordinates,
            region_labels=region_labels,
            quality_scores=quality_scores,
            gene_names=gene_names,
            train_indices=train_indices,
            val_indices=val_indices,
            test_indices=test_indices,
            # Add X and y keys for hyperparameter search compatibility
            X=spot_features,
            y=gene_expression
        )
        
        # Also save a copy for small dataset testing
        small_dataset_dir = os.path.join(output_dir, 'small_dataset', 'features')
        os.makedirs(small_dataset_dir, exist_ok=True)
        
        np.savez(
            os.path.join(small_dataset_dir, 'training_data.npz'),
            spot_features=spot_features[:100],
            subspot_features=subspot_features[:100],
            neighbor_features=neighbor_features[:100],
            gene_expression=gene_expression[:100],
            cell_coordinates=cell_coordinates[:100],
            region_labels=region_labels[:100],
            quality_scores=quality_scores[:100],
            gene_names=gene_names,
            train_indices=np.arange(70),
            val_indices=np.arange(70, 85),
            test_indices=np.arange(85, 100),
            # Add X and y keys for hyperparameter search compatibility
            X=spot_features[:100],
            y=gene_expression[:100]
        )
        
        print(f"Training data prepared and saved to {self.output().path}")
        print(f"Small dataset saved to {os.path.join(small_dataset_dir, 'training_data.npz')}")

#!/usr/bin/env python3
# models/datamodules.py

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import numpy as np
import os

class SpatialDataset(Dataset):
    """
    Dataset for spatial transcriptomics data.
    
    This dataset loads features and gene expression data for training,
    validation, or testing.
    """
    def __init__(self, data_path, indices, neighbor_distance_path=None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the npz file containing the data
            indices: Indices to use (train, val, or test)
            neighbor_distance_path: Path to the npz file containing neighbor distances (optional)
        """
        # Load data
        data = np.load(data_path)
        
        # Extract features and gene expression
        self.spot_features = data['spot_features'][indices]
        self.subspot_features = data['subspot_features'][indices]
        self.neighbor_features = data['neighbor_features'][indices]
        self.gene_expression = data['gene_expression'][indices]
        
        # Load neighbor distances if provided
        self.neighbor_distances = None
        if neighbor_distance_path is not None and os.path.exists(neighbor_distance_path):
            distance_data = np.load(neighbor_distance_path)
            self.neighbor_distances = distance_data['neighbor_distances'][indices]
        
        # Convert to torch tensors
        self.spot_features = torch.tensor(self.spot_features, dtype=torch.float32)
        self.subspot_features = torch.tensor(self.subspot_features, dtype=torch.float32)
        self.neighbor_features = torch.tensor(self.neighbor_features, dtype=torch.float32)
        self.gene_expression = torch.tensor(self.gene_expression, dtype=torch.float32)
        
        if self.neighbor_distances is not None:
            self.neighbor_distances = torch.tensor(self.neighbor_distances, dtype=torch.float32)
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.gene_expression)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
        
        Returns:
            Tuple containing (spot_features, subspot_features, neighbor_features, 
                             neighbor_distances, gene_expression)
        """
        if self.neighbor_distances is not None:
            return (
                self.spot_features[idx],
                self.subspot_features[idx],
                self.neighbor_features[idx],
                self.neighbor_distances[idx],
                self.gene_expression[idx]
            )
        else:
            # Return None for neighbor_distances if not available
            return (
                self.spot_features[idx],
                self.subspot_features[idx],
                self.neighbor_features[idx],
                None,
                self.gene_expression[idx]
            )


class SpatialDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for spatial transcriptomics data.
    
    This module handles loading and preprocessing data for training,
    validation, and testing.
    """
    def __init__(self, config):
        """
        Initialize the data module.
        
        Args:
            config: Configuration dictionary
        """
        super(SpatialDataModule, self).__init__()
        self.config = config
        
        # Get parameters from config
        self.data_path = os.path.join(
            config['output_dir'], 
            'features', 
            'training_data.npz'
        )
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 4)
        
        # Optional neighbor distance path
        self.neighbor_distance_path = config.get('neighbor_distance_path', None)
        
        # Load indices
        data = np.load(self.data_path)
        self.train_indices = data['train_indices']
        self.val_indices = data['val_indices']
        self.test_indices = data['test_indices']
        
        # Set n_genes in config for model initialization
        if 'n_genes' not in config and 'gene_names' in data:
            config['n_genes'] = len(data['gene_names'])
    
    def setup(self, stage=None):
        """
        Set up datasets for training, validation, and testing.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
        if stage == 'fit' or stage is None:
            self.train_dataset = SpatialDataset(
                self.data_path, 
                self.train_indices,
                self.neighbor_distance_path
            )
            self.val_dataset = SpatialDataset(
                self.data_path, 
                self.val_indices,
                self.neighbor_distance_path
            )
        
        if stage == 'test' or stage is None:
            self.test_dataset = SpatialDataset(
                self.data_path, 
                self.test_indices,
                self.neighbor_distance_path
            )
    
    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """Return the test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

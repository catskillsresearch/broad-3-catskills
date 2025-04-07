import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union

class DeepSpotDataset(Dataset):
    """
    Dataset class for DeepSpot model training and evaluation.
    
    This class handles loading and preprocessing of spatial transcriptomics data
    for the unified approach that combines DeepSpot, Tarandros, and LogFC methods.
    """
    
    def __init__(self, data_path: str, split: str = 'train', transform=None):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file (.npz format)
            split: Data split ('train', 'val', or 'test')
            transform: Optional transform to apply to the data
        """
        self.data_path = data_path
        self.split = split
        self.transform = transform
        
        # Load data
        self.data = np.load(data_path, allow_pickle=True)
        
        # Extract data
        self.spot_features = self.data['spot_features']
        self.subspot_features = self.data['subspot_features']
        self.neighbor_features = self.data['neighbor_features']
        self.measured_expressions = self.data['measured_expressions']
        
        # Optional data
        self.subspot_distances = self.data.get('subspot_distances')
        self.neighbor_distances = self.data.get('neighbor_distances')
        self.spatial_coordinates = self.data.get('spatial_coordinates')
        self.unmeasured_expressions = self.data.get('unmeasured_expressions')
        self.cell_types = self.data.get('cell_types')
        self.cell_labels = self.data.get('cell_labels')
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.spot_features)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the sample data
        """
        # Create sample dictionary
        sample = {
            'spot_features': self.spot_features[idx].astype(np.float32),
            'subspot_features': self.subspot_features[idx].astype(np.float32),
            'neighbor_features': self.neighbor_features[idx].astype(np.float32),
            'measured_expressions': self.measured_expressions[idx].astype(np.float32)
        }
        
        # Add optional data if available
        if self.subspot_distances is not None:
            sample['subspot_distances'] = self.subspot_distances[idx].astype(np.float32)
        
        if self.neighbor_distances is not None:
            sample['neighbor_distances'] = self.neighbor_distances[idx].astype(np.float32)
        
        if self.spatial_coordinates is not None:
            sample['spatial_coordinates'] = self.spatial_coordinates[idx].astype(np.float32)
        
        if self.unmeasured_expressions is not None:
            sample['unmeasured_expressions'] = self.unmeasured_expressions[idx].astype(np.float32)
        
        if self.cell_types is not None:
            sample['cell_types'] = self.cell_types[idx].astype(np.int64)
        
        if self.cell_labels is not None:
            sample['cell_labels'] = self.cell_labels[idx]
        
        # Apply transform if available
        if self.transform:
            sample = self.transform(sample)
        
        return sample

class SyntheticDataset(Dataset):
    """
    Dataset class for synthetic data generation and training.
    
    This class provides functionality to create synthetic spatial transcriptomics data
    for pre-training or testing the unified model.
    """
    
    def __init__(self, config: Dict, num_samples: int = 1000):
        """
        Initialize the synthetic dataset.
        
        Args:
            config: Configuration dictionary with dataset parameters
            num_samples: Number of synthetic samples to generate
        """
        self.config = config
        self.num_samples = num_samples
        
        # Dataset parameters
        self.feature_dim = config.get('feature_dim', 512)
        self.n_subspots = config.get('n_subspots', 10)
        self.n_neighbors = config.get('n_neighbors', 8)
        self.n_measured_genes = config.get('n_measured_genes', 460)
        self.n_unmeasured_genes = config.get('n_unmeasured_genes', 18157)
        self.n_cell_types = config.get('n_cell_types', 10)
        
        # Generate synthetic data
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic data for training and testing."""
        # Generate spot features
        self.spot_features = np.random.randn(self.num_samples, self.feature_dim).astype(np.float32)
        
        # Generate subspot features
        self.subspot_features = np.random.randn(self.num_samples, self.n_subspots, self.feature_dim).astype(np.float32)
        
        # Generate neighbor features
        self.neighbor_features = np.random.randn(self.num_samples, self.n_neighbors, self.feature_dim).astype(np.float32)
        
        # Generate subspot distances
        self.subspot_distances = np.abs(np.random.randn(self.num_samples, self.n_subspots)).astype(np.float32)
        
        # Generate neighbor distances
        self.neighbor_distances = np.abs(np.random.randn(self.num_samples, self.n_neighbors)).astype(np.float32)
        
        # Generate spatial coordinates
        self.spatial_coordinates = np.random.randn(self.num_samples, 2).astype(np.float32)
        
        # Generate measured gene expressions
        self.measured_expressions = np.random.randn(self.num_samples, self.n_measured_genes).astype(np.float32)
        
        # Generate unmeasured gene expressions
        self.unmeasured_expressions = np.random.randn(self.num_samples, self.n_unmeasured_genes).astype(np.float32)
        
        # Generate cell types
        self.cell_types = np.random.randint(0, self.n_cell_types, size=self.num_samples).astype(np.int64)
        
        # Generate cell labels (dysplastic or non-dysplastic)
        self.cell_labels = np.random.choice(['dysplastic', 'non_dysplastic'], size=self.num_samples)
    
    def save(self, output_path: str):
        """
        Save the synthetic dataset to a file.
        
        Args:
            output_path: Path to save the dataset
        """
        np.savez(
            output_path,
            spot_features=self.spot_features,
            subspot_features=self.subspot_features,
            neighbor_features=self.neighbor_features,
            subspot_distances=self.subspot_distances,
            neighbor_distances=self.neighbor_distances,
            spatial_coordinates=self.spatial_coordinates,
            measured_expressions=self.measured_expressions,
            unmeasured_expressions=self.unmeasured_expressions,
            cell_types=self.cell_types,
            cell_labels=self.cell_labels
        )
        print(f"Saved synthetic dataset to {output_path}")
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing the sample data
        """
        return {
            'spot_features': self.spot_features[idx],
            'subspot_features': self.subspot_features[idx],
            'neighbor_features': self.neighbor_features[idx],
            'subspot_distances': self.subspot_distances[idx],
            'neighbor_distances': self.neighbor_distances[idx],
            'spatial_coordinates': self.spatial_coordinates[idx],
            'measured_expressions': self.measured_expressions[idx],
            'unmeasured_expressions': self.unmeasured_expressions[idx],
            'cell_types': self.cell_types[idx],
            'cell_labels': self.cell_labels[idx]
        }

class SmallDatasetCreator:
    """
    Class for creating a small dataset from a larger dataset.
    
    This class provides functionality to create a smaller version of a dataset
    for faster development and testing.
    """
    
    def __init__(self, input_path: str, output_path: str, sample_fraction: float = 0.1, random_seed: int = 42):
        """
        Initialize the small dataset creator.
        
        Args:
            input_path: Path to the input dataset
            output_path: Path to save the small dataset
            sample_fraction: Fraction of samples to include in the small dataset
            random_seed: Random seed for reproducibility
        """
        self.input_path = input_path
        self.output_path = output_path
        self.sample_fraction = sample_fraction
        self.random_seed = random_seed
        
        # Set random seed
        np.random.seed(random_seed)
    
    def create(self):
        """Create a small dataset from the input dataset."""
        # Load input data
        data = np.load(self.input_path, allow_pickle=True)
        
        # Get number of samples
        n_samples = len(data['spot_features'])
        
        # Calculate number of samples for small dataset
        n_small_samples = int(n_samples * self.sample_fraction)
        
        # Randomly select samples
        indices = np.random.choice(n_samples, n_small_samples, replace=False)
        
        # Create small dataset
        small_data = {}
        for key in data.keys():
            if isinstance(data[key], np.ndarray) and len(data[key]) == n_samples:
                small_data[key] = data[key][indices]
            else:
                small_data[key] = data[key]
        
        # Save small dataset
        np.savez(self.output_path, **small_data)
        print(f"Created small dataset with {n_small_samples} samples at {self.output_path}")

def create_dataloaders(config: Dict):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration dictionary with dataset parameters
        
    Returns:
        Dictionary containing train, validation, and test data loaders
    """
    import os
    from torch.utils.data import DataLoader, random_split
    
    # Dataset parameters
    data_dir = config.get('data_dir', 'data')
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    val_split = config.get('val_split', 0.1)
    test_split = config.get('test_split', 0.1)
    use_synthetic = config.get('use_synthetic', False)
    use_small_dataset = config.get('use_small_dataset', False)
    
    # Determine dataset path
    if use_synthetic:
        dataset_type = 'synthetic'
    elif use_small_dataset:
        dataset_type = 'small'
    else:
        dataset_type = 'full'
    
    data_path = os.path.join(data_dir, f'{dataset_type}_data.npz')
    
    # Create dataset
    dataset = DeepSpotDataset(data_path=data_path)
    
    # Split dataset
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    test_size = int(dataset_size * test_split)
    train_size = dataset_size - val_size - test_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def create_synthetic_dataset(config: Dict, output_path: str):
    """
    Create a synthetic dataset for training and testing.
    
    Args:
        config: Configuration dictionary with dataset parameters
        output_path: Path to save the synthetic dataset
    """
    # Create synthetic dataset
    dataset = SyntheticDataset(config=config)
    
    # Save dataset
    dataset.save(output_path=output_path)
    
    return dataset

def create_small_dataset(input_path: str, output_path: str, sample_fraction: float = 0.1):
    """
    Create a small dataset from a larger dataset.
    
    Args:
        input_path: Path to the input dataset
        output_path: Path to save the small dataset
        sample_fraction: Fraction of samples to include in the small dataset
    """
    # Create small dataset
    creator = SmallDatasetCreator(
        input_path=input_path,
        output_path=output_path,
        sample_fraction=sample_fraction
    )
    
    # Create dataset
    creator.create()

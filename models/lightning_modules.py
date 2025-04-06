#!/usr/bin/env python3
# models/lightning_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Dict, List, Tuple, Optional


class PhiNetwork(nn.Module):
    """
    Feature transformation network for DeepSpot architecture.
    Transforms input features into a common embedding space.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim * 2)
        self.bn1 = nn.BatchNorm1d(output_dim * 2)
        self.fc2 = nn.Linear(output_dim * 2, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        return x


class RhoNetwork(nn.Module):
    """
    Prediction network for DeepSpot architecture.
    Transforms embeddings into gene expression predictions.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int] = None):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [input_dim * 2, input_dim]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class CellEmbedding(nn.Module):
    """
    Cell embedding module inspired by Tarandros' cell-wise focused approach.
    Optimizes for cell-wise Spearman correlation.
    """
    def __init__(self, input_dim: int, embedding_dim: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, embedding_dim * 2)
        self.bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.fc2 = nn.Linear(embedding_dim * 2, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        return x


class DifferentiableSpearmanLoss(nn.Module):
    """
    Differentiable approximation of Spearman rank correlation loss.
    Uses soft ranking to make the loss differentiable.
    """
    def __init__(self, eps: float = 1e-6, temperature: float = 10.0):
        super().__init__()
        self.eps = eps
        self.temperature = temperature
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute soft ranks
        pred_ranks = self._compute_soft_ranks(pred)
        target_ranks = self._compute_soft_ranks(target)
        
        # Center the ranks
        pred_ranks_centered = pred_ranks - pred_ranks.mean(dim=1, keepdim=True)
        target_ranks_centered = target_ranks - target_ranks.mean(dim=1, keepdim=True)
        
        # Compute correlation
        covariance = (pred_ranks_centered * target_ranks_centered).sum(dim=1)
        pred_std = torch.sqrt((pred_ranks_centered ** 2).sum(dim=1) + self.eps)
        target_std = torch.sqrt((target_ranks_centered ** 2).sum(dim=1) + self.eps)
        
        correlation = covariance / (pred_std * target_std)
        
        # Return loss (1 - correlation)
        return 1 - correlation.mean()
    
    def _compute_soft_ranks(self, x: torch.Tensor) -> torch.Tensor:
        # Compute pairwise differences
        diff = x.unsqueeze(2) - x.unsqueeze(1)
        
        # Apply sigmoid to get soft comparisons
        soft_comparisons = torch.sigmoid(diff * self.temperature)
        
        # Sum over comparisons to get soft ranks
        soft_ranks = soft_comparisons.sum(dim=2) + 0.5
        
        return soft_ranks


class IntegratedSpatialModule(pl.LightningModule):
    """
    PyTorch Lightning module that integrates DeepSpot architecture with
    Tarandros' cell-wise focused approach.
    """
    def __init__(
        self,
        input_dim: int,
        n_genes: int,
        phi_dim: int = 256,
        embedding_dim: int = 512,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        loss_weight_spearman: float = 0.7,
        loss_weight_mse: float = 0.3,
        dropout: float = 0.3
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize DeepSpot components
        self.phi_spot = PhiNetwork(input_dim, phi_dim, dropout)
        self.phi_subspot = PhiNetwork(input_dim, phi_dim, dropout)
        self.phi_neighbor = PhiNetwork(input_dim, phi_dim, dropout)
        
        # Initialize cell-wise focused components
        self.cell_embedding = CellEmbedding(phi_dim * 3, embedding_dim, dropout)
        
        # Initialize prediction head
        self.rho = RhoNetwork(embedding_dim, n_genes)
        
        # Initialize loss functions
        self.mse_loss = nn.MSELoss()
        self.spearman_loss = DifferentiableSpearmanLoss()
        
    def forward(
        self,
        spot_features: torch.Tensor,
        subspot_features: List[torch.Tensor],
        neighbor_features: List[torch.Tensor]
    ) -> torch.Tensor:
        # Process spot features
        spot_embedding = self.phi_spot(spot_features)
        
        # Process sub-spot features
        subspot_embeddings = [self.phi_subspot(f) for f in subspot_features]
        subspot_agg = torch.stack(subspot_embeddings).mean(dim=0)
        
        # Process neighbor features
        neighbor_embeddings = [self.phi_neighbor(f) for f in neighbor_features]
        neighbor_agg = torch.stack(neighbor_embeddings).max(dim=0)[0]
        
        # Concatenate features
        combined_features = torch.cat([spot_embedding, subspot_agg, neighbor_agg], dim=1)
        
        # Apply cell-wise embedding
        cell_embedding = self.cell_embedding(combined_features)
        
        # Predict gene expression
        predictions = self.rho(cell_embedding)
        
        return predictions
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Forward pass
        predictions = self(
            batch['spot_features'],
            batch['subspot_features'],
            batch['neighbor_features']
        )
        targets = batch['gene_expression']
        
        # Compute losses
        mse = self.mse_loss(predictions, targets)
        spearman = self.spearman_loss(predictions, targets)
        
        # Combined loss with cell-wise focus
        loss = (
            self.hparams.loss_weight_mse * mse +
            self.hparams.loss_weight_spearman * spearman
        )
        
        # Log metrics
        self.log('train_loss', loss)
        self.log('train_mse', mse)
        self.log('train_spearman', spearman)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        # Forward pass
        predictions = self(
            batch['spot_features'],
            batch['subspot_features'],
            batch['neighbor_features']
        )
        targets = batch['gene_expression']
        
        # Compute losses
        mse = self.mse_loss(predictions, targets)
        spearman = self.spearman_loss(predictions, targets)
        
        # Combined loss with cell-wise focus
        loss = (
            self.hparams.loss_weight_mse * mse +
            self.hparams.loss_weight_spearman * spearman
        )
        
        # Log metrics
        self.log('val_loss', loss)
        self.log('val_mse', mse)
        self.log('val_spearman', spearman)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        # Forward pass
        predictions = self(
            batch['spot_features'],
            batch['subspot_features'],
            batch['neighbor_features']
        )
        targets = batch['gene_expression']
        
        # Compute losses
        mse = self.mse_loss(predictions, targets)
        spearman = self.spearman_loss(predictions, targets)
        
        # Log metrics
        self.log('test_mse', mse)
        self.log('test_spearman', spearman)
        
        return {
            'predictions': predictions,
            'targets': targets
        }
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        return self(
            batch['spot_features'],
            batch['subspot_features'],
            batch['neighbor_features']
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }


class SpatialDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for spatial transcriptomics data.
    Handles data loading and preparation for training, validation, and testing.
    """
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def setup(self, stage: Optional[str] = None):
        # Load data
        data = np.load(self.data_path)
        
        # Extract features and labels
        self.spot_features = torch.tensor(data['spot_features'], dtype=torch.float32)
        self.subspot_features = torch.tensor(data['subspot_features'], dtype=torch.float32)
        self.neighbor_features = torch.tensor(data['neighbor_features'], dtype=torch.float32)
        self.gene_expression = torch.tensor(data['gene_expression'], dtype=torch.float32)
        
        # Extract indices
        self.train_indices = data['train_indices']
        self.val_indices = data['val_indices']
        self.test_indices = data['test_indices']
        
        # Create datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = self._create_dataset(self.train_indices)
            self.val_dataset = self._create_dataset(self.val_indices)
        
        if stage == 'test' or stage is None:
            self.test_dataset = self._create_dataset(self.test_indices)
        
        if stage == 'predict' or stage is None:
            self.predict_dataset = self._create_dataset(self.test_indices)
    
    def _create_dataset(self, indices):
        # Create a simple dataset from the loaded data
        # In a real implementation, this would be a proper PyTorch Dataset
        return {
            'spot_features': self.spot_features[indices],
            'subspot_features': [self.subspot_features[i] for i in indices],
            'neighbor_features': [self.neighbor_features[i] for i in indices],
            'gene_expression': self.gene_expression[indices]
        }
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

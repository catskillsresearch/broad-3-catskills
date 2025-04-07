#!/usr/bin/env python3
# models/lightning_modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from scipy.stats import spearmanr

from .deepspot_model import DeepSpotModel, spearman_correlation_loss

class DeepSpotLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for DeepSpot model.
    
    This module handles training, validation, and testing of the DeepSpot model,
    as well as logging metrics to W&B.
    """
    def __init__(self, config):
        super(DeepSpotLightningModule, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        
        # Get parameters from config
        self.learning_rate = config.get('learning_rate', 0.0005)
        self.weight_decay = config.get('weight_decay', 0.0001)
        self.loss_weight_spearman = config.get('loss_weight_spearman', 0.7)
        self.loss_weight_mse = config.get('loss_weight_mse', 0.3)
        
        # Initialize model
        self.model = DeepSpotModel(config)
        
    def forward(self, spot_features, subspot_features, neighbor_features, neighbor_distances=None):
        """
        Forward pass of the model.
        
        Args:
            spot_features: Tensor of shape (batch_size, feature_dim)
            subspot_features: Tensor of shape (batch_size, num_subspots, feature_dim)
            neighbor_features: Tensor of shape (batch_size, num_neighbors, feature_dim)
            neighbor_distances: Tensor of shape (batch_size, num_neighbors)
        
        Returns:
            Tensor of shape (batch_size, n_genes) containing predicted gene expression
        """
        return self.model(spot_features, subspot_features, neighbor_features, neighbor_distances)
    
    def training_step(self, batch, batch_idx):
        """
        Training step.
        
        Args:
            batch: Tuple containing (spot_features, subspot_features, neighbor_features, 
                                    neighbor_distances, gene_expression)
            batch_idx: Index of the current batch
        
        Returns:
            Dictionary containing loss and metrics
        """
        # Unpack batch
        spot_features, subspot_features, neighbor_features, neighbor_distances, gene_expression = batch
        
        # Forward pass
        predicted_expression = self(spot_features, subspot_features, neighbor_features, neighbor_distances)
        
        # Compute losses
        mse_loss = F.mse_loss(predicted_expression, gene_expression)
        spearman_loss = spearman_correlation_loss(predicted_expression, gene_expression)
        
        # Combine losses
        loss = (self.loss_weight_mse * mse_loss) + (self.loss_weight_spearman * spearman_loss)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse_loss, on_step=True, on_epoch=True)
        self.log('train_spearman_loss', spearman_loss, on_step=True, on_epoch=True)
        
        return {'loss': loss, 'mse': mse_loss, 'spearman_loss': spearman_loss}
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        Args:
            batch: Tuple containing (spot_features, subspot_features, neighbor_features, 
                                    neighbor_distances, gene_expression)
            batch_idx: Index of the current batch
        
        Returns:
            Dictionary containing loss and metrics
        """
        # Unpack batch
        spot_features, subspot_features, neighbor_features, neighbor_distances, gene_expression = batch
        
        # Forward pass
        predicted_expression = self(spot_features, subspot_features, neighbor_features, neighbor_distances)
        
        # Compute losses
        mse_loss = F.mse_loss(predicted_expression, gene_expression)
        spearman_loss = spearman_correlation_loss(predicted_expression, gene_expression)
        
        # Combine losses
        loss = (self.loss_weight_mse * mse_loss) + (self.loss_weight_spearman * spearman_loss)
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_mse', mse_loss, on_epoch=True)
        self.log('val_spearman_loss', spearman_loss, on_epoch=True)
        
        # Return predictions and targets for epoch end processing
        return {
            'loss': loss,
            'mse': mse_loss,
            'spearman_loss': spearman_loss,
            'preds': predicted_expression.detach().cpu(),
            'targets': gene_expression.detach().cpu()
        }
    
    def validation_epoch_end(self, outputs):
        """
        Process validation epoch results.
        
        Args:
            outputs: List of dictionaries returned by validation_step
        """
        # Concatenate predictions and targets from all batches
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        
        # Convert to numpy for spearmanr calculation
        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()
        
        # Calculate cell-wise Spearman correlation
        cell_wise_corrs = []
        for i in range(preds_np.shape[0]):
            corr, _ = spearmanr(preds_np[i], targets_np[i])
            if not np.isnan(corr):
                cell_wise_corrs.append(corr)
        
        cell_wise_spearman = np.mean(cell_wise_corrs)
        
        # Calculate gene-wise Spearman correlation
        gene_wise_corrs = []
        for i in range(preds_np.shape[1]):
            corr, _ = spearmanr(preds_np[:, i], targets_np[:, i])
            if not np.isnan(corr):
                gene_wise_corrs.append(corr)
        
        gene_wise_spearman = np.mean(gene_wise_corrs)
        
        # Log metrics
        self.log('val_cell_wise_spearman', cell_wise_spearman, on_epoch=True)
        self.log('val_gene_wise_spearman', gene_wise_spearman, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        """
        Test step.
        
        Args:
            batch: Tuple containing (spot_features, subspot_features, neighbor_features, 
                                    neighbor_distances, gene_expression)
            batch_idx: Index of the current batch
        
        Returns:
            Dictionary containing loss and metrics
        """
        # Unpack batch
        spot_features, subspot_features, neighbor_features, neighbor_distances, gene_expression = batch
        
        # Forward pass
        predicted_expression = self(spot_features, subspot_features, neighbor_features, neighbor_distances)
        
        # Compute losses
        mse_loss = F.mse_loss(predicted_expression, gene_expression)
        spearman_loss = spearman_correlation_loss(predicted_expression, gene_expression)
        
        # Combine losses
        loss = (self.loss_weight_mse * mse_loss) + (self.loss_weight_spearman * spearman_loss)
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_mse', mse_loss, on_epoch=True)
        self.log('test_spearman_loss', spearman_loss, on_epoch=True)
        
        # Return predictions and targets for epoch end processing
        return {
            'loss': loss,
            'mse': mse_loss,
            'spearman_loss': spearman_loss,
            'preds': predicted_expression.detach().cpu(),
            'targets': gene_expression.detach().cpu()
        }
    
    def test_epoch_end(self, outputs):
        """
        Process test epoch results.
        
        Args:
            outputs: List of dictionaries returned by test_step
        """
        # Concatenate predictions and targets from all batches
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        
        # Convert to numpy for spearmanr calculation
        preds_np = all_preds.numpy()
        targets_np = all_targets.numpy()
        
        # Calculate cell-wise Spearman correlation
        cell_wise_corrs = []
        for i in range(preds_np.shape[0]):
            corr, _ = spearmanr(preds_np[i], targets_np[i])
            if not np.isnan(corr):
                cell_wise_corrs.append(corr)
        
        cell_wise_spearman = np.mean(cell_wise_corrs)
        
        # Calculate gene-wise Spearman correlation
        gene_wise_corrs = []
        for i in range(preds_np.shape[1]):
            corr, _ = spearmanr(preds_np[:, i], targets_np[:, i])
            if not np.isnan(corr):
                gene_wise_corrs.append(corr)
        
        gene_wise_spearman = np.mean(gene_wise_corrs)
        
        # Log metrics
        self.log('test_cell_wise_spearman', cell_wise_spearman, on_epoch=True)
        self.log('test_gene_wise_spearman', gene_wise_spearman, on_epoch=True)
        
        # Save detailed results for later analysis
        self.test_results = {
            'predictions': preds_np,
            'targets': targets_np,
            'cell_wise_spearman': cell_wise_spearman,
            'gene_wise_spearman': gene_wise_spearman,
            'cell_wise_corrs': cell_wise_corrs,
            'gene_wise_corrs': gene_wise_corrs
        }
    
    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers.
        
        Returns:
            Dictionary containing optimizer and scheduler
        """
        optimizer = Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = ReduceLROnPlateau(
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

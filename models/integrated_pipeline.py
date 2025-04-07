#!/usr/bin/env python3
# models/integrated_pipeline.py

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import pytorch_lightning as pl

# Import local modules
from models.deepspot_model import DeepSpotModel, spearman_correlation_loss
from models.tarandros_model import TarandrosModel, cell_wise_spearman_loss
from models.logfc_gene_ranking import LogFCGeneRanking, LogFCBasedFeatureSelection

class IntegratedPipeline:
    """
    Integrated pipeline that combines DeepSpot (Crunch 1), Tarandros (Crunch 2),
    and logFC (Crunch 3) approaches for spatial transcriptomics analysis.
    """
    def __init__(self, config: Dict):
        """
        Initialize the integrated pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize DeepSpot model for Crunch 1 (predicting measured genes)
        self.deepspot_model = DeepSpotModel(config)
        
        # Initialize Tarandros model for Crunch 2 (predicting unmeasured genes)
        self.tarandros_model = TarandrosModel(config)
        
        # Initialize logFC gene ranking for Crunch 3
        self.logfc_ranker = LogFCGeneRanking(config)
        
        # Paths for saving results
        self.output_dir = config.get('output_dir', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
        
    def run_crunch1(self, 
                   spot_features: np.ndarray,
                   subspot_features: np.ndarray,
                   neighbor_features: np.ndarray,
                   neighbor_distances: np.ndarray,
                   gene_expression: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Run Crunch 1 to predict measured gene expression using DeepSpot model.
        
        Args:
            spot_features: Spot-level features (n_spots, feature_dim)
            subspot_features: Sub-spot features (n_spots, n_subspots, feature_dim)
            neighbor_features: Neighbor features (n_spots, n_neighbors, feature_dim)
            neighbor_distances: Neighbor distances (n_spots, n_neighbors)
            gene_expression: Ground truth gene expression (n_spots, n_measured_genes)
            
        Returns:
            Predicted measured gene expression (n_spots, n_measured_genes)
        """
        print("Running Crunch 1: Predicting measured gene expression with DeepSpot...")
        
        # Convert numpy arrays to torch tensors
        spot_features_tensor = torch.tensor(spot_features, dtype=torch.float32, device=self.device)
        subspot_features_tensor = torch.tensor(subspot_features, dtype=torch.float32, device=self.device)
        neighbor_features_tensor = torch.tensor(neighbor_features, dtype=torch.float32, device=self.device)
        neighbor_distances_tensor = torch.tensor(neighbor_distances, dtype=torch.float32, device=self.device)
        
        # Move model to device
        self.deepspot_model = self.deepspot_model.to(self.device)
        
        # Set model to evaluation mode
        self.deepspot_model.eval()
        
        # Make predictions
        with torch.no_grad():
            predicted_expression = self.deepspot_model(
                spot_features_tensor,
                subspot_features_tensor,
                neighbor_features_tensor,
                neighbor_distances_tensor
            )
        
        # Convert predictions to numpy
        predicted_expression_np = predicted_expression.cpu().numpy()
        
        # Evaluate if ground truth is provided
        if gene_expression is not None:
            from scipy.stats import spearmanr
            
            # Compute gene-wise Spearman correlation
            gene_wise_corrs = []
            for i in range(gene_expression.shape[1]):
                corr, _ = spearmanr(predicted_expression_np[:, i], gene_expression[:, i])
                if not np.isnan(corr):
                    gene_wise_corrs.append(corr)
            
            mean_gene_wise_spearman = np.mean(gene_wise_corrs)
            print(f"Crunch 1 - Mean gene-wise Spearman correlation: {mean_gene_wise_spearman:.4f}")
            
            # Compute cell-wise Spearman correlation
            cell_wise_corrs = []
            for i in range(gene_expression.shape[0]):
                corr, _ = spearmanr(predicted_expression_np[i], gene_expression[i])
                if not np.isnan(corr):
                    cell_wise_corrs.append(corr)
            
            mean_cell_wise_spearman = np.mean(cell_wise_corrs)
            print(f"Crunch 1 - Mean cell-wise Spearman correlation: {mean_cell_wise_spearman:.4f}")
        
        # Save predictions
        np.save(os.path.join(self.output_dir, 'crunch1_predictions.npy'), predicted_expression_np)
        
        return predicted_expression_np
    
    def run_crunch2(self, 
                   spot_features: np.ndarray,
                   measured_gene_expression: np.ndarray,
                   neighbor_features: Optional[np.ndarray] = None,
                   neighbor_distances: Optional[np.ndarray] = None,
                   reference_data: Optional[Dict] = None) -> np.ndarray:
        """
        Run Crunch 2 to predict unmeasured gene expression using Tarandros model.
        
        Args:
            spot_features: Spot-level features (n_spots, feature_dim)
            measured_gene_expression: Measured gene expression (n_spots, n_measured_genes)
            neighbor_features: Neighbor features (n_spots, n_neighbors, feature_dim)
            neighbor_distances: Neighbor distances (n_spots, n_neighbors)
            reference_data: Reference data for scRNA-seq
            
        Returns:
            Predicted unmeasured gene expression (n_spots, n_unmeasured_genes)
        """
        print("Running Crunch 2: Predicting unmeasured gene expression with Tarandros...")
        
        # Convert numpy arrays to torch tensors
        spot_features_tensor = torch.tensor(spot_features, dtype=torch.float32, device=self.device)
        measured_expression_tensor = torch.tensor(measured_gene_expression, dtype=torch.float32, device=self.device)
        
        # Convert neighbor data if provided
        if neighbor_features is not None and neighbor_distances is not None:
            neighbor_features_tensor = torch.tensor(neighbor_features, dtype=torch.float32, device=self.device)
            neighbor_distances_tensor = torch.tensor(neighbor_distances, dtype=torch.float32, device=self.device)
        else:
            neighbor_features_tensor = None
            neighbor_distances_tensor = None
        
        # Set reference data if provided
        if reference_data is not None:
            measured_ref = torch.tensor(reference_data['measured'], dtype=torch.float32, device=self.device)
            unmeasured_ref = torch.tensor(reference_data['unmeasured'], dtype=torch.float32, device=self.device)
            self.tarandros_model.set_reference_data(measured_ref, unmeasured_ref)
        
        # Move model to device
        self.tarandros_model = self.tarandros_model.to(self.device)
        
        # Set model to evaluation mode
        self.tarandros_model.eval()
        
        # Make predictions
        with torch.no_grad():
            predicted_expression = self.tarandros_model(
                spot_features_tensor,
                measured_expression_tensor,
                neighbor_features_tensor,
                neighbor_distances_tensor
            )
        
        # Convert predictions to numpy
        predicted_expression_np = predicted_expression.cpu().numpy()
        
        # Save predictions
        np.save(os.path.join(self.output_dir, 'crunch2_predictions.npy'), predicted_expression_np)
        
        return predicted_expression_np
    
    def run_crunch3(self, 
                   gene_expression: np.ndarray,
                   cell_labels: np.ndarray,
                   gene_names: List[str]) -> pd.DataFrame:
        """
        Run Crunch 3 to rank genes using logFC method.
        
        Args:
            gene_expression: Gene expression (n_cells, n_genes)
            cell_labels: Cell labels (n_cells), 1 for dysplastic, 0 for non-dysplastic
            gene_names: List of gene names
            
        Returns:
            DataFrame with gene rankings
        """
        print("Running Crunch 3: Ranking genes with logFC method...")
        
        # Split expression data by cell type
        dysplastic_mask = cell_labels == 1
        non_dysplastic_mask = cell_labels == 0
        
        dysplastic_expression = gene_expression[dysplastic_mask]
        non_dysplastic_expression = gene_expression[non_dysplastic_mask]
        
        print(f"Number of dysplastic cells: {dysplastic_expression.shape[0]}")
        print(f"Number of non-dysplastic cells: {non_dysplastic_expression.shape[0]}")
        
        # Rank genes
        gene_rankings = self.logfc_ranker.rank_genes(
            dysplastic_expression,
            non_dysplastic_expression,
            gene_names
        )
        
        # Save rankings
        rankings_path = os.path.join(self.output_dir, 'gene_rankings.csv')
        self.logfc_ranker.save_rankings(rankings_path)
        
        # Create visualization
        viz_path = os.path.join(self.output_dir, 'gene_rankings_visualization.png')
        self.logfc_ranker.visualize_rankings(viz_path)
        
        return gene_rankings
    
    def run_full_pipeline(self, 
                         spot_features: np.ndarray,
                         subspot_features: np.ndarray,
                         neighbor_features: np.ndarray,
                         neighbor_distances: np.ndarray,
                         measured_gene_expression: Optional[np.ndarray] = None,
                         cell_labels: Optional[np.ndarray] = None,
                         measured_gene_names: Optional[List[str]] = None,
                         unmeasured_gene_names: Optional[List[str]] = None,
                         reference_data: Optional[Dict] = None) -> Dict:
        """
        Run the full integrated pipeline.
        
        Args:
            spot_features: Spot-level features (n_spots, feature_dim)
            subspot_features: Sub-spot features (n_spots, n_subspots, feature_dim)
            neighbor_features: Neighbor features (n_spots, n_neighbors, feature_dim)
            neighbor_distances: Neighbor distances (n_spots, n_neighbors)
            measured_gene_expression: Measured gene expression (n_spots, n_measured_genes)
            cell_labels: Cell labels (n_spots), 1 for dysplastic, 0 for non-dysplastic
            measured_gene_names: List of measured gene names
            unmeasured_gene_names: List of unmeasured gene names
            reference_data: Reference data for scRNA-seq
            
        Returns:
            Dictionary with results from all three crunches
        """
        print("Running full integrated pipeline...")
        
        # Run Crunch 1
        predicted_measured_expression = self.run_crunch1(
            spot_features,
            subspot_features,
            neighbor_features,
            neighbor_distances,
            measured_gene_expression
        )
        
        # Use predicted expression if ground truth is not available
        if measured_gene_expression is None:
            measured_gene_expression = predicted_measured_expression
        
        # Run Crunch 2
        predicted_unmeasured_expression = self.run_crunch2(
            spot_features,
            measured_gene_expression,
            neighbor_features,
            neighbor_distances,
            reference_data
        )
        
        # Combine measured and unmeasured gene expression
        all_gene_expression = np.concatenate([measured_gene_expression, predicted_unmeasured_expression], axis=1)
        all_gene_names = measured_gene_names + unmeasured_gene_names if measured_gene_names and unmeasured_gene_names else None
        
        # Run Crunch 3 if cell labels are provided
        gene_rankings = None
        if cell_labels is not None and all_gene_names is not None:
            gene_rankings = self.run_crunch3(
                all_gene_expression,
                cell_labels,
                all_gene_names
            )
        
        # Return results
        results = {
            'crunch1_predictions': predicted_measured_expression,
            'crunch2_predictions': predicted_unmeasured_expression,
            'combined_expression': all_gene_expression,
            'gene_rankings': gene_rankings
        }
        
        return results


class IntegratedLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for the integrated pipeline.
    """
    def __init__(self, config: Dict):
        """
        Initialize the integrated lightning module.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize DeepSpot model for Crunch 1
        self.deepspot_model = DeepSpotModel(config)
        
        # Initialize Tarandros model for Crunch 2
        self.tarandros_model = TarandrosModel(config)
        
        # Loss weights
        self.gene_wise_weight = config.get('gene_wise_weight', 0.3)
        self.cell_wise_weight = config.get('cell_wise_weight', 0.7)
        
    def forward(self, 
               spot_features: torch.Tensor,
               subspot_features: torch.Tensor,
               neighbor_features: torch.Tensor,
               neighbor_distances: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the integrated model.
        
        Args:
            spot_features: Spot-level features (batch_size, feature_dim)
            subspot_features: Sub-spot features (batch_size, n_subspots, feature_dim)
            neighbor_features: Neighbor features (batch_size, n_neighbors, feature_dim)
            neighbor_distances: Neighbor distances (batch_size, n_neighbors)
            
        Returns:
            Dictionary with predictions
        """
        # Predict measured genes with DeepSpot
        measured_predictions = self.deepspot_model(
            spot_features,
            subspot_features,
            neighbor_features,
            neighbor_distances
        )
        
        # Predict unmeasured genes with Tarandros
        unmeasured_predictions = self.tarandros_model(
            spot_features,
            measured_predictions,
            neighbor_features,
            neighbor_distances
        )
        
        return {
            'measured_predictions': measured_predictions,
            'unmeasured_predictions': unmeasured_predictions
        }
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        # Extract batch data
        spot_features = batch['spot_features']
        subspot_features = batch['subspot_features']
        neighbor_features = batch['neighbor_features']
        neighbor_distances = batch['neighbor_distances']
        measured_expression = batch['measured_expression']
        unmeasured_expression = batch.get('unmeasured_expression', None)
        
        # Forward pass
        predictions = self(
            spot_features,
            subspot_features,
            neighbor_features,
            neighbor_distances
        )
        
        measured_predictions = predictions['measured_predictions']
        
        # Compute loss for measured genes
        gene_wise_loss = spearman_correlation_loss(measured_predictions, measured_expression)
        cell_wise_loss = cell_wise_spearman_loss(measured_predictions, measured_expression)
        
        # Combine losses with weights
        measured_loss = self.gene_wise_weight * gene_wise_loss + self.cell_wise_weight * cell_wise_loss
        
        # Compute loss for unmeasured genes if available
        unmeasured_loss = 0.0
        if unmeasured_expression is not None:
            unmeasured_predictions = predictions['unmeasured_predictions']
            unmeasured_gene_wise_loss = spearman_correlation_loss(unmeasured_predictions, unmeasured_expression)
            unmeasured_cell_wise_loss = cell_wise_spearman_loss(unmeasured_predictions, unmeasured_expression)
            unmeasured_loss = self.gene_wise_weight * unmeasured_gene_wise_loss + self.cell_wise_weight * unmeasured_cell_wise_loss
        
        # Total loss
        total_loss = measured_loss + unmeasured_loss
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_measured_loss', measured_loss, on_step=True, on_epoch=True)
        self.log('train_gene_wise_loss', gene_wise_loss, on_step=True, on_epoch=True)
        self.log('train_cell_wise_loss', cell_wise_loss, on_step=True, on_epoch=True)
        
        if unmeasured_expression is not None:
            self.log('train_unmeasured_loss', unmeasured_loss, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
        """
        # Extract batch data
        spot_features = batch['spot_features']
        subspot_features = batch['subspot_features']
        neighbor_features = batch['neighbor_features']
        neighbor_distances = batch['neighbor_distances']
        measured_expression = batch['measured_expression']
        unmeasured_expression = batch.get('unmeasured_expression', None)
        
        # Forward pass
        predictions = self(
            spot_features,
            subspot_features,
            neighbor_features,
            neighbor_distances
        )
        
        measured_predictions = predictions['measured_predictions']
        
        # Compute loss for measured genes
        gene_wise_loss = spearman_correlation_loss(measured_predictions, measured_expression)
        cell_wise_loss = cell_wise_spearman_loss(measured_predictions, measured_expression)
        
        # Combine losses with weights
        measured_loss = self.gene_wise_weight * gene_wise_loss + self.cell_wise_weight * cell_wise_loss
        
        # Compute loss for unmeasured genes if available
        unmeasured_loss = 0.0
        if unmeasured_expression is not None:
            unmeasured_predictions = predictions['unmeasured_predictions']
            unmeasured_gene_wise_loss = spearman_correlation_loss(unmeasured_predictions, unmeasured_expression)
            unmeasured_cell_wise_loss = cell_wise_spearman_loss(unmeasured_predictions, unmeasured_expression)
            unmeasured_loss = self.gene_wise_weight * unmeasured_gene_wise_loss + self.cell_wise_weight * unmeasured_cell_wise_loss
        
        # Total loss
        total_loss = measured_loss + unmeasured_loss
        
        # Log metrics
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_measured_loss', measured_loss, on_epoch=True)
        self.log('val_gene_wise_loss', gene_wise_loss, on_epoch=True)
        self.log('val_cell_wise_loss', cell_wise_loss, on_epoch=True)
        
        if unmeasured_expression is not None:
            self.log('val_unmeasured_loss', unmeasured_loss, on_epoch=True)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Test step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
        """
        # Extract batch data
        spot_features = batch['spot_features']
        subspot_features = batch['subspot_features']
        neighbor_features = batch['neighbor_features']
        neighbor_distances = batch['neighbor_distances']
        measured_expression = batch['measured_expression']
        unmeasured_expression = batch.get('unmeasured_expression', None)
        
        # Forward pass
        predictions = self(
            spot_features,
            subspot_features,
            neighbor_features,
            neighbor_distances
        )
        
        measured_predictions = predictions['measured_predictions']
        
        # Compute metrics for measured genes
        gene_wise_loss = spearman_correlation_loss(measured_predictions, measured_expression)
        cell_wise_loss = cell_wise_spearman_loss(measured_predictions, measured_expression)
        
        # Log metrics
        self.log('test_gene_wise_loss', gene_wise_loss, on_epoch=True)
        self.log('test_cell_wise_loss', cell_wise_loss, on_epoch=True)
        
        # Compute metrics for unmeasured genes if available
        if unmeasured_expression is not None:
            unmeasured_predictions = predictions['unmeasured_predictions']
            unmeasured_gene_wise_loss = spearman_correlation_loss(unmeasured_predictions, unmeasured_expression)
            unmeasured_cell_wise_loss = cell_wise_spearman_loss(unmeasured_predictions, unmeasured_expression)
            
            self.log('test_unmeasured_gene_wise_loss', unmeasured_gene_wise_loss, on_epoch=True)
            self.log('test_unmeasured_cell_wise_loss', unmeasured_cell_wise_loss, on_epoch=True)
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Predict step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
            
        Returns:
            Dictionary with predictions
        """
        # Extract batch data
        spot_features = batch['spot_features']
        subspot_features = batch['subspot_features']
        neighbor_features = batch['neighbor_features']
        neighbor_distances = batch['neighbor_distances']
        
        # Forward pass
        predictions = self(
            spot_features,
            subspot_features,
            neighbor_features,
            neighbor_distances
        )
        
        return predictions
    
    def configure_optimizers(self):
        """
        Configure optimizers.
        
        Returns:
            Optimizer
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 0.0001)
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

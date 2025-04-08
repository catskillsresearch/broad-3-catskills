# Fix for tkinter errors - added by direct_tkinter_fix.py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend that doesn't require tkinter

#!/usr/bin/env python3
# scripts/run_hyperparameter_search.py

import argparse
import json
import os
import numpy as np
import luigi
import yaml
import wandb
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline tasks
from pipeline.pipeline_data_preparation import PrepareTrainingData, EnsureDirectories

class SimpleModel(pl.LightningModule):
    """
    A simple PyTorch Lightning model for the hyperparameter search.
    """
    def __init__(self, input_dim, output_dim, phi_dim=256, dropout=0.2, 
                 learning_rate=0.001, weight_decay=1e-5, loss_weight_spearman=0.7):
        super().__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.phi_network = nn.Sequential(
            nn.Linear(input_dim, phi_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(phi_dim, phi_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.rho_network = nn.Sequential(
            nn.Linear(phi_dim, phi_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(phi_dim, output_dim)
        )
        
        # Loss function weights
        self.loss_weight_spearman = loss_weight_spearman
        self.loss_weight_mse = 1.0 - loss_weight_spearman
        
    def forward(self, x):
        phi_features = self.phi_network(x)
        predictions = self.rho_network(phi_features)
        return predictions
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # MSE Loss
        mse_loss = nn.functional.mse_loss(y_hat, y)
        
        # Spearman correlation loss (approximated)
        # Convert to ranks
        y_ranks = torch.argsort(torch.argsort(y, dim=1), dim=1).float()
        y_hat_ranks = torch.argsort(torch.argsort(y_hat, dim=1), dim=1).float()
        
        # Compute correlation loss (1 - correlation)
        spearman_loss = nn.functional.mse_loss(y_hat_ranks, y_ranks)
        
        # Combined loss
        loss = self.loss_weight_mse * mse_loss + self.loss_weight_spearman * spearman_loss
        
        # Log metrics
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_mse', mse_loss, on_step=False, on_epoch=True)
        self.log('train_spearman_loss', spearman_loss, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        
        # MSE Loss
        mse_loss = nn.functional.mse_loss(y_hat, y)
        
        # Calculate cell-wise Spearman correlation
        cell_wise_corr = self._calculate_cell_wise_spearman(y_hat, y)
        
        # Calculate gene-wise Spearman correlation
        gene_wise_corr = self._calculate_gene_wise_spearman(y_hat, y)
        
        # Log metrics
        self.log('val_mse', mse_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_cell_wise_spearman', cell_wise_corr, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_gene_wise_spearman', gene_wise_corr, on_step=False, on_epoch=True)
        
        # Store predictions for visualization
        if batch_idx == 0:
            self.validation_predictions = y_hat.detach().cpu().numpy()
            self.validation_targets = y.detach().cpu().numpy()
            self.validation_inputs = x.detach().cpu().numpy()
        
        return {'val_loss': mse_loss, 'val_cell_wise_spearman': cell_wise_corr}
    
    def on_validation_epoch_end(self):
        # Create and log visualizations if we have validation data
        if hasattr(self, 'validation_predictions') and hasattr(self, 'validation_targets'):
            self._create_and_log_visualizations()
    
    def _calculate_cell_wise_spearman(self, y_hat, y):
        """Calculate mean cell-wise Spearman correlation."""
        # Move to CPU for numpy operations
        y_hat_np = y_hat.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        # Calculate correlation for each cell (row)
        correlations = []
        for i in range(y_np.shape[0]):
            # Use numpy's corrcoef which is faster than scipy's spearmanr for this case
            # Convert to ranks first
            y_ranks = np.argsort(np.argsort(y_np[i]))
            y_hat_ranks = np.argsort(np.argsort(y_hat_np[i]))
            
            # Calculate Pearson correlation of ranks (equivalent to Spearman)
            corr_matrix = np.corrcoef(y_ranks, y_hat_ranks)
            corr = corr_matrix[0, 1]
            
            # Handle NaN values
            if np.isnan(corr):
                corr = 0.0
                
            correlations.append(corr)
        
        # Return mean correlation
        return np.mean(correlations)
    
    def _calculate_gene_wise_spearman(self, y_hat, y):
        """Calculate mean gene-wise Spearman correlation."""
        # Move to CPU for numpy operations
        y_hat_np = y_hat.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        
        # Calculate correlation for each gene (column)
        correlations = []
        for i in range(y_np.shape[1]):
            # Use numpy's corrcoef which is faster than scipy's spearmanr for this case
            # Convert to ranks first
            y_ranks = np.argsort(np.argsort(y_np[:, i]))
            y_hat_ranks = np.argsort(np.argsort(y_hat_np[:, i]))
            
            # Calculate Pearson correlation of ranks (equivalent to Spearman)
            corr_matrix = np.corrcoef(y_ranks, y_hat_ranks)
            corr = corr_matrix[0, 1]
            
            # Handle NaN values
            if np.isnan(corr):
                corr = 0.0
                
            correlations.append(corr)
        
        # Return mean correlation
        return np.mean(correlations)
    
    def _create_and_log_visualizations(self):
        """Create and log visualizations to W&B."""
        # Create output directory for visualizations
        output_dir = self.logger.experiment.dir
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Get data
        predictions = self.validation_predictions
        targets = self.validation_targets
        inputs = self.validation_inputs
        
        # Create a dictionary to store visualization paths
        vis_paths = {}
        
        # 1. PCA Visualization
        try:
            pca_vis_path = self._create_pca_visualization(predictions, targets, vis_dir)
            vis_paths['pca_visualization'] = pca_vis_path
        except Exception as e:
            print(f"Warning: Could not create PCA visualization: {e}")
            vis_paths['pca_visualization'] = None
        
        # 2. t-SNE Visualization
        try:
            tsne_vis_path = self._create_tsne_visualization(predictions, targets, vis_dir)
            vis_paths['tsne_visualization'] = tsne_vis_path
        except Exception as e:
            print(f"Warning: Could not create t-SNE visualization: {e}")
            vis_paths['tsne_visualization'] = None
        
        # 3. UMAP Visualization
        try:
            umap_vis_path = self._create_umap_visualization(predictions, targets, vis_dir)
            vis_paths['umap_visualization'] = umap_vis_path
        except Exception as e:
            print(f"Warning: Could not create UMAP visualization: {e}")
            vis_paths['umap_visualization'] = None
        
        # 4. Prediction vs Ground Truth Heatmap
        try:
            heatmap_vis_path = self._create_heatmap_visualization(predictions, targets, vis_dir)
            vis_paths['heatmap_visualization'] = heatmap_vis_path
        except Exception as e:
            print(f"Warning: Could not create heatmap visualization: {e}")
            vis_paths['heatmap_visualization'] = None
        
        # Log images to W&B (only the ones that were successfully created)
        log_dict = {}
        for key, path in vis_paths.items():
            if path is not None:
                log_dict[key] = wandb.Image(path)
        
        if log_dict:
            self.logger.experiment.log(log_dict)
        
        # Create HTML visualization page with available visualizations
        valid_paths = [path for path in vis_paths.values() if path is not None]
        if valid_paths:
            self._create_html_visualization_page(valid_paths, vis_dir)
    
    def _create_pca_visualization(self, predictions, targets, vis_dir):
        """Create PCA visualization of predictions vs targets."""
        # Apply PCA
        pca = PCA(n_components=2)
        pca_pred = pca.fit_transform(predictions)
        pca_target = pca.transform(targets)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot predictions
        plt.scatter(pca_pred[:, 0], pca_pred[:, 1], c='blue', alpha=0.5, label='Predictions')
        
        # Plot targets
        plt.scatter(pca_target[:, 0], pca_target[:, 1], c='red', alpha=0.5, label='Ground Truth')
        
        # Add arrows connecting corresponding points
        for i in range(len(pca_pred)):
            plt.arrow(pca_target[i, 0], pca_target[i, 1], 
                     pca_pred[i, 0] - pca_target[i, 0], 
                     pca_pred[i, 1] - pca_target[i, 1], 
                     color='gray', alpha=0.3, width=0.001, head_width=0.01)
        
        plt.title('PCA: Predictions vs Ground Truth')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        save_path = os.path.join(vis_dir, 'pca_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _create_tsne_visualization(self, predictions, targets, vis_dir):
        """Create t-SNE visualization of predictions vs targets."""
        # Check if we have enough samples for t-SNE
        n_samples = len(predictions) + len(targets)
        
        # Calculate appropriate perplexity (should be less than n_samples)
        # Recommended range is 5-50, but must be less than n_samples
        perplexity = min(30, max(5, n_samples // 5))
        
        # If we still don't have enough samples, raise an exception
        if perplexity >= n_samples:
            raise ValueError(f"Not enough samples for t-SNE visualization. Need at least 6 samples, got {n_samples}.")
        
        # Apply t-SNE with adjusted perplexity
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        combined_data = np.vstack([predictions, targets])
        tsne_result = tsne.fit_transform(combined_data)
        
        # Split results back into predictions and targets
        tsne_pred = tsne_result[:len(predictions)]
        tsne_target = tsne_result[len(predictions):]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot predictions
        plt.scatter(tsne_pred[:, 0], tsne_pred[:, 1], c='blue', alpha=0.5, label='Predictions')
        
        # Plot targets
        plt.scatter(tsne_target[:, 0], tsne_target[:, 1], c='red', alpha=0.5, label='Ground Truth')
        
        # Add arrows connecting corresponding points
        for i in range(len(tsne_pred)):
            plt.arrow(tsne_target[i, 0], tsne_target[i, 1], 
                     tsne_pred[i, 0] - tsne_target[i, 0], 
                     tsne_pred[i, 1] - tsne_target[i, 1], 
                     color='gray', alpha=0.3, width=0.001, head_width=0.01)
        
        plt.title(f't-SNE (perplexity={perplexity}): Predictions vs Ground Truth')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        save_path = os.path.join(vis_dir, 'tsne_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _create_umap_visualization(self, predictions, targets, vis_dir):
        """Create UMAP visualization of predictions vs targets."""
        # Check if we have enough samples for UMAP
        n_samples = len(predictions) + len(targets)
        
        # UMAP requires at least 2 samples
        if n_samples < 2:
            raise ValueError(f"Not enough samples for UMAP visualization. Need at least 2 samples, got {n_samples}.")
        
        # Adjust n_neighbors parameter based on dataset size
        n_neighbors = min(15, max(2, n_samples // 5))
        
        # Apply UMAP with adjusted parameters
        reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors)
        combined_data = np.vstack([predictions, targets])
        umap_result = reducer.fit_transform(combined_data)
        
        # Split results back into predictions and targets
        umap_pred = umap_result[:len(predictions)]
        umap_target = umap_result[len(predictions):]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot predictions
        plt.scatter(umap_pred[:, 0], umap_pred[:, 1], c='blue', alpha=0.5, label='Predictions')
        
        # Plot targets
        plt.scatter(umap_target[:, 0], umap_target[:, 1], c='red', alpha=0.5, label='Ground Truth')
        
        # Add arrows connecting corresponding points
        for i in range(len(umap_pred)):
            plt.arrow(umap_target[i, 0], umap_target[i, 1], 
                     umap_pred[i, 0] - umap_target[i, 0], 
                     umap_pred[i, 1] - umap_target[i, 1], 
                     color='gray', alpha=0.3, width=0.001, head_width=0.01)
        
        plt.title(f'UMAP (n_neighbors={n_neighbors}): Predictions vs Ground Truth')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        save_path = os.path.join(vis_dir, 'umap_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _create_heatmap_visualization(self, predictions, targets, vis_dir):
        """Create heatmap visualization comparing predictions to targets."""
        # Limit to a subset of cells and genes for better visualization
        max_cells = min(10, predictions.shape[0])
        max_genes = min(50, predictions.shape[1])
        
        # Select a subset of cells and genes
        cell_indices = np.linspace(0, predictions.shape[0]-1, max_cells, dtype=int)
        gene_indices = np.linspace(0, predictions.shape[1]-1, max_genes, dtype=int)
        
        pred_subset = predictions[cell_indices][:, gene_indices]
        target_subset = targets[cell_indices][:, gene_indices]
        
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot ground truth
        sns.heatmap(target_subset, ax=ax1, cmap='viridis', cbar_kws={'label': 'Expression'})
        ax1.set_title('Ground Truth')
        ax1.set_xlabel('Genes')
        ax1.set_ylabel('Cells')
        
        # Plot predictions
        sns.heatmap(pred_subset, ax=ax2, cmap='viridis', cbar_kws={'label': 'Expression'})
        ax2.set_title('Predictions')
        ax2.set_xlabel('Genes')
        ax2.set_ylabel('Cells')
        
        # Calculate correlations for each cell
        correlations = []
        for i in range(max_cells):
            corr_matrix = np.corrcoef(target_subset[i], pred_subset[i])
            corr = corr_matrix[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        # Add overall correlation as a figure title
        if correlations:
            mean_corr = np.mean(correlations)
            plt.suptitle(f'Gene Expression Comparison (Mean Cell Correlation: {mean_corr:.3f})', fontsize=16)
        else:
            plt.suptitle('Gene Expression Comparison', fontsize=16)
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(vis_dir, 'heatmap_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _create_html_visualization_page(self, image_paths, vis_dir):
        """Create an HTML page with all visualizations."""
        html_content = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Model Visualization Results</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }
                h1 {
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                }
                h2 {
                    color: #2980b9;
                    margin-top: 30px;
                }
                .visualization {
                    margin: 30px 0;
                    padding: 20px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                img {
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 20px 0;
                    border: 1px solid #ddd;
                }
                .metrics {
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #e7f5fe;
                    border-left: 4px solid #3498db;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f2f2f2;
                }
            </style>
        </head>
        <body>
            <h1>Model Visualization Results</h1>
            
            <div class="metrics">
                <h2>Performance Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """
        
        # Add metrics if available
        metrics = {}
        if hasattr(self, 'trainer') and hasattr(self.trainer, 'callback_metrics'):
            for key, value in self.trainer.callback_metrics.items():
                if isinstance(value, torch.Tensor):
                    metrics[key] = value.item()
                else:
                    metrics[key] = value
        
        for key, value in metrics.items():
            html_content += f"""
                    <tr>
                        <td>{key}</td>
                        <td>{value:.4f}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        """
        
        # Add visualizations
        for path in image_paths:
            if path is None:
                continue
                
            vis_type = os.path.basename(path).replace('_visualization.png', '')
            
            # Create a more readable title
            if vis_type == 'pca':
                title = 'PCA Visualization'
                description = """
                    <p>Principal Component Analysis (PCA) reduces the high-dimensional gene expression data to two dimensions 
                    while preserving global variance. Blue points represent model predictions, red points represent ground truth values, 
                    and gray arrows connect corresponding points. Shorter arrows indicate better predictions.</p>
                """
            elif vis_type == 'tsne':
                title = 't-SNE Visualization'
                description = """
                    <p>t-Distributed Stochastic Neighbor Embedding (t-SNE) reduces dimensionality while preserving local structure. 
                    Blue points represent model predictions, red points represent ground truth values. 
                    Proximity of corresponding points indicates prediction accuracy.</p>
                """
            elif vis_type == 'umap':
                title = 'UMAP Visualization'
                description = """
                    <p>Uniform Manifold Approximation and Projection (UMAP) preserves both local and global structure. 
                    Blue points represent model predictions, red points represent ground truth values. 
                    Similarity in overall patterns indicates how well the model captures data structure.</p>
                """
            elif vis_type == 'heatmap':
                title = 'Gene Expression Heatmap'
                description = """
                    <p>This heatmap compares predicted gene expression with ground truth for selected cells. 
                    The left panel shows ground truth expression, while the right panel shows predicted expression. 
                    Correlation values provide a quantitative measure of prediction accuracy.</p>
                """
            else:
                title = f'{vis_type.title()} Visualization'
                description = ""
            
            # Get relative path for the image
            rel_path = os.path.basename(path)
            
            html_content += f"""
            <div class="visualization">
                <h2>{title}</h2>
                {description}
                <img src="{rel_path}" alt="{title}">
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Write HTML file
        html_path = os.path.join(vis_dir, 'visualizations.html')
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return html_path
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optimizer

class HyperparameterSearch(luigi.Task):
    """
    Luigi task for running hyperparameter search.
    """
    config_path = luigi.Parameter(description="Path to the main configuration file")
    sweep_config_path = luigi.Parameter(default="config/hyperparameter_search.yaml", description="Path to the sweep configuration file")
    count = luigi.IntParameter(default=10, description="Number of runs to execute")
    use_small_dataset = luigi.BoolParameter(default=False, description="Whether to use the small dataset")
    seed = luigi.IntParameter(default=42, description="Random seed for reproducibility")
    
    
    def ensure_directories(self):
        # Ensure all necessary directories exist.
        # Load config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get output directory
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        # Create directories for hyperparameter search
        directories = [
            output_dir,
            os.path.join(output_dir, 'sweeps'),
            os.path.join(output_dir, 'small_dataset'),
            os.path.join(output_dir, 'small_dataset', 'features')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Ensured directory exists: {directory}")
    
    
    def ensure_directories(self):
        # Ensure all necessary directories exist.
        # Load config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Get output directory
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        # Create directories for hyperparameter search
        directories = [
            output_dir,
            os.path.join(output_dir, 'sweeps'),
            os.path.join(output_dir, 'small_dataset'),
            os.path.join(output_dir, 'small_dataset', 'features')
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"Ensured directory exists: {directory}")
    
    def requires(self):
        # If using small dataset, require the small dataset preparation task
        if self.use_small_dataset:
            return PrepareTrainingData(config_path="config/small_dataset_config.yaml")
        else:
            return PrepareTrainingData(config_path=self.config_path)
    
    def output(self):
        return luigi.LocalTarget(f"output/sweeps/sweep_results.yaml")
    
    def run(self):
        # Ensure directories exist
        self.ensure_directories()

        # Ensure directories exist
        self.ensure_directories()

        # Load the sweep configuration
        with open(self.sweep_config_path, 'r') as f:
            sweep_config = yaml.safe_load(f)
        
        # Determine the data path based on whether we're using the small dataset
        if self.use_small_dataset:
            data_path = "output/small_dataset/features/training_data.npz"
            print(f"Using training data from: {data_path}")
        else:
            data_path = "output/features/training_data.npz"
            print(f"Using training data from: {data_path}")
        
        # Extract only the method, metric, and parameters sections for W&B
        clean_sweep_config = {
            "method": sweep_config.get("method", "random"),
            "metric": sweep_config.get("metric", {"name": "val_cell_wise_spearman", "goal": "maximize"}),
            "parameters": sweep_config.get("parameters", {})
        }
        
        # Add seed and small_dataset flag to parameters
        if "parameters" not in clean_sweep_config:
            clean_sweep_config["parameters"] = {}
        
        clean_sweep_config["parameters"]["seed"] = {"value": self.seed}
        clean_sweep_config["parameters"]["use_small_dataset"] = {"value": self.use_small_dataset}
        
        # Fix log_uniform distributions if present
        for param_name, param_config in clean_sweep_config["parameters"].items():
            if param_config.get("distribution") == "log_uniform":
                # Convert to log_uniform_values
                min_val = param_config.get("min", 0)
                max_val = param_config.get("max", 0)
                param_config["distribution"] = "log_uniform_values"
                param_config["min"] = np.exp(min_val)
                param_config["max"] = np.exp(max_val)
        
        # Initialize wandb
        sweep_id = wandb.sweep(clean_sweep_config, project="spatial-transcriptomics")
        print(f"Create sweep with ID: {sweep_id}")
        print(f"Sweep URL: {wandb.run.get_sweep_url() if wandb.run else 'https://wandb.ai/username/spatial-transcriptomics/sweeps/' + sweep_id}")
        
        # Create a wrapper function for the wandb agent
        def train_model_with_config():
            # Initialize a new wandb run
            with wandb.init() as run:
                # Get hyperparameters from wandb
                config = wandb.config
                
                # Set random seeds for reproducibility
                seed = config.get('seed', 42)
                torch.manual_seed(seed)
                np.random.seed(seed)
                
                # Load data
                if config.get('use_small_dataset', False):
                    data_path = "output/small_dataset/features/training_data.npz"
                else:
                    data_path = "output/features/training_data.npz"
                
                try:
                    data = np.load(data_path, allow_pickle=True)
                    
                    # Try to get X and y directly
                    try:
                        X = data['X']
                        y = data['y']
                    except KeyError:
                        print("X and y keys not found in data file, constructing from features...")
                        
                        # Construct X and y from the available features
                        spot_features = data['spot_features']
                        subspot_features = data['subspot_features']
                        neighbor_features = data['neighbor_features']
                        gene_expression = data['gene_expression']
                        
                        # Create X as concatenated features
                        n_samples = len(spot_features)
                        feature_dim = spot_features.shape[1]
                        n_subspots = subspot_features.shape[1]
                        n_neighbors = neighbor_features.shape[1]
                        
                        X = np.concatenate([
                            spot_features,
                            subspot_features.reshape(n_samples, n_subspots * feature_dim),
                            neighbor_features.reshape(n_samples, n_neighbors * feature_dim)
                        ], axis=1)
                        
                        # y is the gene expression
                        y = gene_expression
                        
                        print(f"Successfully constructed X with shape {X.shape} and y with shape {y.shape}")
                        
                except FileNotFoundError:
                    print(f"Error: Training data not found at {data_path}")
                    print("Make sure to run the data preparation step first.")
                    return
                except Exception as e:
                    print(f"Error loading data: {e}")
                    print("Creating placeholder data for testing...")
                    
                    # Create placeholder data for testing
                    n_samples = 10
                    feature_dim = 512
                    n_genes = 100
                    
                    X = np.random.rand(n_samples, feature_dim * (1 + 4 + 6))  # spot + subspot + neighbor
                    y = np.random.rand(n_samples, n_genes)
                    print(f"Created placeholder X with shape {X.shape} and y with shape {y.shape}")
                
                # Split data into train and validation sets
                n_samples = X.shape[0]
                indices = np.random.permutation(n_samples)
                train_size = int(0.8 * n_samples)
                
                train_indices = indices[:train_size]
                val_indices = indices[train_size:]
                
                X_train, y_train = X[train_indices], y[train_indices]
                X_val, y_val = X[val_indices], y[val_indices]
                
                # Convert to PyTorch tensors
                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
                
                # Create datasets and dataloaders
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
                
                batch_size = config.get('batch_size', 64)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
                
                # Create model
                input_dim = X.shape[1]
                output_dim = y.shape[1]
                
                model = SimpleModel(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    phi_dim=config.get('phi_dim', 256),
                    dropout=config.get('dropout', 0.2),
                    learning_rate=config.get('learning_rate', 0.001),
                    weight_decay=config.get('weight_decay', 1e-5),
                    loss_weight_spearman=config.get('loss_weight_spearman', 0.7)
                )
                
                # Create callbacks
                checkpoint_callback = ModelCheckpoint(
                    dirpath=f"output/models/{run.id}",
                    filename="model-{epoch:02d}-{val_cell_wise_spearman:.4f}",
                    monitor="val_cell_wise_spearman",
                    mode="max",
                    save_top_k=1
                )
                
                early_stop_callback = EarlyStopping(
                    monitor="val_cell_wise_spearman",
                    mode="max",
                    patience=5,
                    verbose=True
                )
                
                # Create trainer
                trainer = pl.Trainer(
                    max_epochs=20,  # Train for 20 epochs
                    logger=WandbLogger(),
                    callbacks=[checkpoint_callback, early_stop_callback],
                    enable_progress_bar=True,
                    log_every_n_steps=1
                )
                
                # Train model
                trainer.fit(model, train_loader, val_loader)
                
                # Log best model path
                if checkpoint_callback.best_model_path:
                    wandb.log({"best_model_path": checkpoint_callback.best_model_path})
        
        # Run the wandb agent
        wandb.agent(sweep_id, function=train_model_with_config, count=self.count)
        
        # Save sweep results
        os.makedirs(os.path.dirname(self.output().path), exist_ok=True)
        with open(self.output().path, 'w') as f:
            yaml.dump({
                'sweep_id': sweep_id,
                'config_path': self.config_path,
                'sweep_config_path': self.sweep_config_path,
                'count': self.count,
                'use_small_dataset': self.use_small_dataset,
                'seed': self.seed
            }, f)

def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter search for the model")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration file")
    parser.add_argument("--sweep_config", type=str, default="config/hyperparameter_search.yaml", help="Path to the sweep configuration file")
    parser.add_argument("--count", type=int, default=10, help="Number of runs to execute")
    parser.add_argument("--small_dataset", action="store_true", help="Use the small dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Run the Luigi task
    luigi.build([
        HyperparameterSearch(
            config_path=args.config,
            sweep_config_path=args.sweep_config,
            count=args.count,
            use_small_dataset=args.small_dataset,
            seed=args.seed
        )
    ], local_scheduler=True)

if __name__ == "__main__":
    main()

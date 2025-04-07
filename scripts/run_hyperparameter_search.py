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
        
        # 1. PCA Visualization
        pca_vis_path = self._create_pca_visualization(predictions, targets, vis_dir)
        
        # 2. t-SNE Visualization
        tsne_vis_path = self._create_tsne_visualization(predictions, targets, vis_dir)
        
        # 3. UMAP Visualization
        umap_vis_path = self._create_umap_visualization(predictions, targets, vis_dir)
        
        # 4. Prediction vs Ground Truth Heatmap
        heatmap_vis_path = self._create_heatmap_visualization(predictions, targets, vis_dir)
        
        # Log images to W&B
        self.logger.experiment.log({
            "pca_visualization": wandb.Image(pca_vis_path),
            "tsne_visualization": wandb.Image(tsne_vis_path),
            "umap_visualization": wandb.Image(umap_vis_path),
            "heatmap_visualization": wandb.Image(heatmap_vis_path)
        })
        
        # Create HTML visualization page
        self._create_html_visualization_page(
            [pca_vis_path, tsne_vis_path, umap_vis_path, heatmap_vis_path],
            vis_dir
        )
    
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
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
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
        
        plt.title('t-SNE: Predictions vs Ground Truth')
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
        # Apply UMAP
        reducer = umap.UMAP(random_state=42)
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
        
        plt.title('UMAP: Predictions vs Ground Truth')
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
        """Create heatmap visualization of predictions vs targets."""
        # Sample a subset of genes and cells for better visualization
        n_cells = min(50, predictions.shape[0])
        n_genes = min(50, predictions.shape[1])
        
        pred_subset = predictions[:n_cells, :n_genes]
        target_subset = targets[:n_cells, :n_genes]
        
        # Calculate correlation matrix
        corr_matrix = np.zeros((n_cells, 2))
        for i in range(n_cells):
            # Calculate Spearman correlation for each cell
            y_ranks = np.argsort(np.argsort(target_subset[i]))
            y_hat_ranks = np.argsort(np.argsort(pred_subset[i]))
            corr_matrix[i, 0] = np.corrcoef(y_ranks, y_hat_ranks)[0, 1]
            corr_matrix[i, 1] = i  # Cell index
        
        # Sort by correlation
        corr_matrix = corr_matrix[corr_matrix[:, 0].argsort()]
        
        # Select cells with highest, lowest, and middle correlations
        selected_indices = np.concatenate([
            corr_matrix[:5, 1].astype(int),  # Lowest correlation
            corr_matrix[n_cells//2-2:n_cells//2+3, 1].astype(int),  # Middle correlation
            corr_matrix[-5:, 1].astype(int)  # Highest correlation
        ])
        
        # Create figure with subplots
        fig, axes = plt.subplots(len(selected_indices), 2, figsize=(12, 3*len(selected_indices)))
        
        for i, idx in enumerate(selected_indices):
            # Plot ground truth
            sns.heatmap(target_subset[idx].reshape(1, -1), ax=axes[i, 0], cmap='viridis', 
                       cbar=False, xticklabels=False, yticklabels=['Cell ' + str(idx)])
            
            # Plot prediction
            sns.heatmap(pred_subset[idx].reshape(1, -1), ax=axes[i, 1], cmap='viridis', 
                       cbar=True, xticklabels=False, yticklabels=['Cell ' + str(idx)])
            
            # Add correlation value
            corr = np.corrcoef(
                np.argsort(np.argsort(target_subset[idx])), 
                np.argsort(np.argsort(pred_subset[idx]))
            )[0, 1]
            axes[i, 0].set_title(f'Ground Truth (Cell {idx})')
            axes[i, 1].set_title(f'Prediction (Corr: {corr:.3f})')
        
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
                    margin: 20px 0;
                    text-align: center;
                }
                .visualization img {
                    max-width: 100%;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .description {
                    margin: 15px 0;
                    text-align: left;
                    max-width: 800px;
                    margin-left: auto;
                    margin-right: auto;
                }
                .metrics {
                    background-color: #f8f9fa;
                    border: 1px solid #e9ecef;
                    border-radius: 4px;
                    padding: 15px;
                    margin: 20px 0;
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
                    <tr>
                        <td>Cell-wise Spearman Correlation</td>
                        <td id="cell-wise-spearman">Loading...</td>
                    </tr>
                    <tr>
                        <td>Gene-wise Spearman Correlation</td>
                        <td id="gene-wise-spearman">Loading...</td>
                    </tr>
                    <tr>
                        <td>Mean Squared Error</td>
                        <td id="mse">Loading...</td>
                    </tr>
                </table>
            </div>
        """
        
        # Add PCA visualization
        pca_path = os.path.basename(image_paths[0])
        html_content += f"""
            <h2>PCA Visualization</h2>
            <div class="visualization">
                <img src="{pca_path}" alt="PCA Visualization">
            </div>
            <div class="description">
                <p>This visualization shows the projection of gene expression data onto the first two principal components. 
                Blue points represent model predictions, while red points represent ground truth values. 
                Gray arrows connect corresponding points, with shorter arrows indicating better predictions.</p>
            </div>
        """
        
        # Add t-SNE visualization
        tsne_path = os.path.basename(image_paths[1])
        html_content += f"""
            <h2>t-SNE Visualization</h2>
            <div class="visualization">
                <img src="{tsne_path}" alt="t-SNE Visualization">
            </div>
            <div class="description">
                <p>This t-SNE plot reduces the high-dimensional gene expression data to two dimensions while preserving local structure.
                Blue points represent model predictions, while red points represent ground truth values.
                The proximity of corresponding blue and red points indicates prediction accuracy.</p>
            </div>
        """
        
        # Add UMAP visualization
        umap_path = os.path.basename(image_paths[2])
        html_content += f"""
            <h2>UMAP Visualization</h2>
            <div class="visualization">
                <img src="{umap_path}" alt="UMAP Visualization">
            </div>
            <div class="description">
                <p>UMAP provides another dimensionality reduction technique that often preserves both local and global structure better than t-SNE.
                Blue points represent model predictions, while red points represent ground truth values.
                The similarity in the overall patterns between predictions and ground truth indicates how well the model captures the data structure.</p>
            </div>
        """
        
        # Add heatmap visualization
        heatmap_path = os.path.basename(image_paths[3])
        html_content += f"""
            <h2>Gene Expression Heatmap</h2>
            <div class="visualization">
                <img src="{heatmap_path}" alt="Heatmap Visualization">
            </div>
            <div class="description">
                <p>These heatmaps compare predicted gene expression (right) with ground truth (left) for selected cells.
                Cells are chosen to represent the range of prediction quality, from lowest to highest correlation.
                The similarity in patterns between pairs of heatmaps indicates prediction accuracy for individual cells.</p>
            </div>
        """
        
        # Close HTML
        html_content += """
            <script>
                // This would be populated with actual metrics in a real implementation
                document.getElementById('cell-wise-spearman').textContent = 'PLACEHOLDER';
                document.getElementById('gene-wise-spearman').textContent = 'PLACEHOLDER';
                document.getElementById('mse').textContent = 'PLACEHOLDER';
            </script>
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
    Luigi task for running hyperparameter search using Weights & Biases.
    This task depends on PrepareTrainingData to ensure the training data exists.
    """
    config_path = luigi.Parameter(description="Path to configuration file")
    sweep_config_path = luigi.Parameter(default="config/hyperparameter_search.yaml", 
                                       description="Path to sweep configuration file")
    count = luigi.IntParameter(default=10, description="Number of runs to execute")
    use_small_dataset = luigi.BoolParameter(default=False, description="Use small dataset for hyperparameter search")
    seed = luigi.IntParameter(default=42, description="Random seed for reproducibility")
    
    def requires(self):
        """
        Define dependencies for this task.
        The hyperparameter search requires the training data to be prepared.
        """
        # If using small dataset, use small_dataset_config.yaml
        if self.use_small_dataset:
            config_path = 'config/small_dataset_config.yaml'
        else:
            config_path = self.config_path
            
        return PrepareTrainingData(config_path=config_path)
    
    def output(self):
        """
        Define the output target for this task.
        The hyperparameter search produces a sweep results file.
        """
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config['output_dir']
        sweeps_dir = os.path.join(output_dir, 'sweeps')
        Path(sweeps_dir).mkdir(parents=True, exist_ok=True)
        
        return luigi.LocalTarget(os.path.join(sweeps_dir, 'sweep_results.yaml'))
    
    def run(self):
        """
        Run the hyperparameter search using Weights & Biases.
        """
        # If using small dataset, use small_dataset_config.yaml
        if self.use_small_dataset:
            config_path = 'config/small_dataset_config.yaml'
        else:
            config_path = self.config_path
            
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load sweep config
        with open(self.sweep_config_path, 'r') as f:
            sweep_config = yaml.safe_load(f)
        
        # Get the path to the training data
        output_dir = config['output_dir']
        features_dir = os.path.join(output_dir, 'features')
        training_data_path = os.path.join(features_dir, 'training_data.npz')
        
        print(f"Using training data from: {training_data_path}")
        
        # Ensure sweep config doesn't contain extraneous properties
        # Only keep the method, metric, and parameters sections
        clean_sweep_config = {
            'method': sweep_config.get('method', 'bayes'),
            'metric': sweep_config.get('metric', {'name': 'val_cell_wise_spearman', 'goal': 'maximize'}),
            'parameters': sweep_config.get('parameters', {})
        }
        
        # Add small_dataset flag and seed to sweep config parameters
        clean_sweep_config['parameters']['use_small_dataset'] = {'value': self.use_small_dataset}
        clean_sweep_config['parameters']['seed'] = {'value': self.seed}
        
        # Create output directories
        hyperparams_dir = os.path.join(features_dir, 'hyperparameters')
        Path(hyperparams_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        wandb.login()
        
        # Create sweep
        sweep_id = wandb.sweep(clean_sweep_config, project='spatial-transcriptomics')
        print(f"Created sweep with ID: {sweep_id}")
        print(f"View sweep at: https://wandb.ai/username/spatial-transcriptomics/sweeps/{sweep_id}")
        
        # Define the training function
        def train_model_with_config(config=None):
            with wandb.init(config=config) as run:
                # Get hyperparameters from wandb
                config = wandb.config
                
                # Load training data
                data = np.load(training_data_path, allow_pickle=True)
                
                # Extract features and targets
                features = data['spot_features']
                targets = data['gene_expression']
                train_indices = data['train_indices']
                val_indices = data['val_indices']
                test_indices = data['test_indices']
                
                # Convert to PyTorch tensors
                X_train = torch.tensor(features[train_indices], dtype=torch.float32)
                y_train = torch.tensor(targets[train_indices], dtype=torch.float32)
                X_val = torch.tensor(features[val_indices], dtype=torch.float32)
                y_val = torch.tensor(targets[val_indices], dtype=torch.float32)
                
                # Create data loaders
                train_dataset = TensorDataset(X_train, y_train)
                val_dataset = TensorDataset(X_val, y_val)
                
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=int(config.batch_size), 
                    shuffle=True
                )
                val_loader = DataLoader(
                    val_dataset, 
                    batch_size=int(config.batch_size), 
                    shuffle=False
                )
                
                # Create model
                model = SimpleModel(
                    input_dim=features.shape[1],
                    output_dim=targets.shape[1],
                    phi_dim=int(config.phi_dim),
                    dropout=float(config.dropout),
                    learning_rate=float(config.learning_rate),
                    weight_decay=float(config.weight_decay),
                    loss_weight_spearman=float(config.loss_weight_spearman)
                )
                
                # Create callbacks
                checkpoint_callback = ModelCheckpoint(
                    monitor='val_cell_wise_spearman',
                    mode='max',
                    save_top_k=1,
                    filename='best-{epoch:02d}-{val_cell_wise_spearman:.4f}'
                )
                
                early_stop_callback = EarlyStopping(
                    monitor='val_cell_wise_spearman',
                    patience=5,
                    mode='max'
                )
                
                # Create logger
                wandb_logger = WandbLogger(experiment=run)
                
                # Create trainer
                trainer = pl.Trainer(
                    max_epochs=20,  # Run for 20 epochs as requested
                    logger=wandb_logger,
                    callbacks=[checkpoint_callback, early_stop_callback],
                    enable_progress_bar=True,
                    log_every_n_steps=5
                )
                
                # Train model
                trainer.fit(model, train_loader, val_loader)
                
                # Get best validation metrics
                best_val_cell_wise_spearman = checkpoint_callback.best_model_score.item()
                
                # Log best metrics
                wandb.log({
                    'best_val_cell_wise_spearman': best_val_cell_wise_spearman
                })
                
                # Create visualizations directory in output
                vis_output_dir = os.path.join(output_dir, 'visualizations', run.id)
                Path(vis_output_dir).mkdir(parents=True, exist_ok=True)
                
                # Copy visualization files from wandb run directory to output directory
                wandb_run_dir = wandb_logger.experiment.dir
                wandb_vis_dir = os.path.join(wandb_run_dir, 'visualizations')
                
                if os.path.exists(wandb_vis_dir):
                    for file in os.listdir(wandb_vis_dir):
                        if file.endswith('.png') or file.endswith('.html'):
                            src = os.path.join(wandb_vis_dir, file)
                            dst = os.path.join(vis_output_dir, file)
                            try:
                                import shutil
                                shutil.copy2(src, dst)
                                print(f"Copied visualization file: {file}")
                            except Exception as e:
                                print(f"Error copying file {file}: {e}")
                
                return best_val_cell_wise_spearman
        
        # Run sweep
        wandb.agent(sweep_id, function=train_model_with_config, count=self.count)
        
        # Get best run
        api = wandb.Api()
        sweep = api.sweep(f"username/spatial-transcriptomics/sweeps/{sweep_id}")
        runs = sorted(sweep.runs, key=lambda run: run.summary.get('best_val_cell_wise_spearman', 0), reverse=True)
        
        if runs:
            best_run = runs[0]
            best_config = {k: v for k, v in best_run.config.items() if not k.startswith('_')}
            best_metrics = {
                'best_val_cell_wise_spearman': best_run.summary.get('best_val_cell_wise_spearman', 0),
                'best_val_gene_wise_spearman': best_run.summary.get('val_gene_wise_spearman', 0),
                'best_val_mse': best_run.summary.get('val_mse', 0)
            }
            
            # Save best config
            with open(os.path.join(hyperparams_dir, 'optimal_params.json'), 'w') as f:
                json.dump(best_config, f, indent=2)
            
            # Save sweep results
            sweep_results = {
                'sweep_id': sweep_id,
                'best_run_id': best_run.id,
                'best_config': best_config,
                'best_metrics': best_metrics,
                'visualization_dir': os.path.join(output_dir, 'visualizations', best_run.id)
            }
            
            with self.output().open('w') as f:
                yaml.dump(sweep_results, f, default_flow_style=False)
            
            print(f"Hyperparameter search complete. Best run: {best_run.id}")
            print(f"Best cell-wise Spearman: {best_metrics['best_val_cell_wise_spearman']:.4f}")
            print(f"Best gene-wise Spearman: {best_metrics.get('best_val_gene_wise_spearman', 'N/A')}")
            print(f"Best MSE: {best_metrics.get('best_val_mse', 'N/A')}")
            print(f"Optimal parameters saved to {os.path.join(hyperparams_dir, 'optimal_params.json')}")
            print(f"Sweep results saved to {self.output().path}")
            print(f"Visualizations saved to {os.path.join(output_dir, 'visualizations', best_run.id)}")
        else:
            print("No runs found in sweep.")
            # Create an empty output file to satisfy Luigi
            with self.output().open('w') as f:
                yaml.dump({'error': 'No runs found in sweep'}, f)

def run_hyperparameter_search(config_path, sweep_config_path=None, count=10, use_small_dataset=False, seed=42):
    """
    Run the hyperparameter search using Luigi to manage dependencies.
    
    Parameters:
    - config_path: Path to configuration file
    - sweep_config_path: Path to sweep configuration file
    - count: Number of runs to execute
    - use_small_dataset: Whether to use the small dataset
    - seed: Random seed for reproducibility
    """
    # Run the hyperparameter search task
    task = HyperparameterSearch(
        config_path=config_path,
        sweep_config_path=sweep_config_path if sweep_config_path else "config/hyperparameter_search.yaml",
        count=count,
        use_small_dataset=use_small_dataset,
        seed=seed
    )
    
    # Build the task
    luigi.build([task], local_scheduler=True)
    
    print(f"Hyperparameter search completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run hyperparameter search for spatial transcriptomics model')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to configuration file')
    parser.add_argument('--sweep_config', type=str, help='Path to sweep configuration file')
    parser.add_argument('--count', type=int, default=10, help='Number of runs to execute')
    parser.add_argument('--small_dataset', action='store_true', help='Use small dataset for hyperparameter search')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    run_hyperparameter_search(
        config_path=args.config,
        sweep_config_path=args.sweep_config,
        count=args.count,
        use_small_dataset=args.small_dataset,
        seed=args.seed
    )

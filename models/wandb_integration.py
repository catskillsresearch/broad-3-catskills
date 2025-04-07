#!/usr/bin/env python3
# models/wandb_integration.py

import os
import yaml
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

def setup_wandb_integration(config, experiment_name='default_experiment'):
    """
    Set up Weights & Biases integration for the DeepSpot model.
    
    Args:
        config: Configuration dictionary
        experiment_name: Name of the experiment
    
    Returns:
        Tuple containing (wandb_logger, callbacks)
    """
    # Initialize W&B logger
    wandb_logger = WandbLogger(
        project=config.get('wandb_project', 'spatial-transcriptomics'),
        name=experiment_name,
        config=config,
        log_model=True
    )
    
    # Define model checkpoint callback
    models_dir = os.path.join(config['output_dir'], 'models', experiment_name)
    os.makedirs(models_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=models_dir,
        filename='model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True
    )
    
    # Define early stopping callback
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config.get('patience', 10),
        mode='min',
        verbose=True
    )
    
    # Define learning rate monitor callback
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Return logger and callbacks
    callbacks = [checkpoint_callback, early_stop_callback, lr_monitor]
    
    return wandb_logger, callbacks

def log_model_summary(model, wandb_logger):
    """
    Log model summary to W&B.
    
    Args:
        model: PyTorch model
        wandb_logger: W&B logger
    """
    # Log model architecture
    wandb_logger.experiment.config.update({"model_summary": str(model)})

def log_evaluation_results(results, experiment_name='default_experiment', config=None):
    """
    Log evaluation results to W&B.
    
    Args:
        results: Dictionary containing evaluation results
        experiment_name: Name of the experiment
        config: Configuration dictionary
    """
    # Initialize W&B run
    if config is None:
        config = {}
    
    wandb.init(
        project=config.get('wandb_project', 'spatial-transcriptomics'),
        name=f"{experiment_name}_evaluation",
        config=config,
        reinit=True
    )
    
    # Log metrics
    wandb.log({
        'cell_wise_spearman': results['cell_wise_spearman'],
        'gene_wise_spearman': results['gene_wise_spearman'],
        'mse': results.get('mse', 0.0)
    })
    
    # Create and log histograms
    if 'cell_wise_corrs' in results:
        wandb.log({
            'cell_wise_corr_histogram': wandb.Histogram(results['cell_wise_corrs'])
        })
    
    if 'gene_wise_corrs' in results:
        wandb.log({
            'gene_wise_corr_histogram': wandb.Histogram(results['gene_wise_corrs'])
        })
    
    # Finish the run
    wandb.finish()

def create_wandb_plots(predictions, targets, gene_names=None, cell_indices=None):
    """
    Create and log plots to W&B.
    
    Args:
        predictions: Predicted gene expression values
        targets: Ground truth gene expression values
        gene_names: List of gene names
        cell_indices: List of cell indices
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import spearmanr
    
    # Create scatter plot for a random gene
    gene_idx = np.random.randint(0, predictions.shape[1])
    gene_name = gene_names[gene_idx] if gene_names is not None else f"Gene {gene_idx}"
    
    plt.figure(figsize=(10, 6))
    plt.scatter(targets[:, gene_idx], predictions[:, gene_idx], alpha=0.5)
    plt.xlabel('True Expression')
    plt.ylabel('Predicted Expression')
    plt.title(f'True vs Predicted Expression for {gene_name}')
    
    # Calculate correlation
    corr, _ = spearmanr(targets[:, gene_idx], predictions[:, gene_idx])
    plt.annotate(f'Spearman r = {corr:.4f}', xy=(0.05, 0.95), xycoords='axes fraction')
    
    # Log to W&B
    wandb.log({f"gene_scatter_{gene_name}": wandb.Image(plt)})
    plt.close()
    
    # Create heatmap of top genes
    # Calculate gene-wise correlations
    gene_corrs = []
    for i in range(predictions.shape[1]):
        corr, _ = spearmanr(predictions[:, i], targets[:, i])
        if not np.isnan(corr):
            gene_corrs.append((i, corr))
    
    # Sort by correlation
    gene_corrs.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 20 genes
    top_genes = gene_corrs[:20]
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    # Get indices and correlations
    indices = [x[0] for x in top_genes]
    corrs = [x[1] for x in top_genes]
    
    # Get gene names if available
    labels = [gene_names[i] if gene_names is not None else f"Gene {i}" for i in indices]
    
    # Plot bar chart
    plt.barh(range(len(labels)), corrs, tick_label=labels)
    plt.xlabel('Spearman Correlation')
    plt.ylabel('Gene')
    plt.title('Top 20 Genes by Prediction Accuracy')
    plt.tight_layout()
    
    # Log to W&B
    wandb.log({"top_genes_chart": wandb.Image(plt)})
    plt.close()

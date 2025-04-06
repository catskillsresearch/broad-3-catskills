#!/usr/bin/env python3
# scripts/run_hyperparameter_search.py

import os
import argparse
import yaml
import wandb
import luigi
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Import local modules
# In a real implementation, these would be properly imported
# from models.lightning_modules import IntegratedSpatialModule, SpatialDataModule
# from pipeline.tasks.model_training import TrainModel, EvaluateModel

def parse_args():
    parser = argparse.ArgumentParser(description='Run hyperparameter search for spatial transcriptomics model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--sweep_config', type=str, default='config/hyperparameter_search.yaml',
                        help='Path to sweep configuration file')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of runs to execute')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def train_model_with_config():
    """
    Training function to be called by wandb.agent.
    Uses the hyperparameters from wandb.config.
    """
    # Initialize wandb run
    run = wandb.init()
    
    # Get hyperparameters from wandb
    config = wandb.config
    
    # Load data
    data_path = config.data_path
    data = np.load(data_path)
    
    # Extract features and labels
    spot_features = torch.tensor(data['spot_features'], dtype=torch.float32)
    subspot_features = torch.tensor(data['subspot_features'], dtype=torch.float32)
    neighbor_features = torch.tensor(data['neighbor_features'], dtype=torch.float32)
    gene_expression = torch.tensor(data['gene_expression'], dtype=torch.float32)
    
    # Extract indices
    train_indices = data['train_indices']
    val_indices = data['val_indices']
    
    # Create data module
    # In a real implementation, this would use the actual SpatialDataModule
    # data_module = SpatialDataModule(data_path, batch_size=config.batch_size)
    
    # Initialize model
    # In a real implementation, this would use the actual IntegratedSpatialModule
    # model = IntegratedSpatialModule(
    #     input_dim=spot_features.shape[1],
    #     n_genes=gene_expression.shape[1],
    #     phi_dim=config.phi_dim,
    #     embedding_dim=config.embedding_dim,
    #     learning_rate=config.learning_rate,
    #     weight_decay=config.weight_decay,
    #     loss_weight_spearman=config.loss_weight_spearman,
    #     loss_weight_mse=1.0 - config.loss_weight_spearman,
    #     dropout=config.dropout
    # )
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config.wandb_project,
        log_model=True
    )
    
    # Define callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.output_dir, 'models', f'sweep_{run.id}'),
        filename='model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=config.patience,
        mode='min'
    )
    
    # Initialize trainer
    # trainer = pl.Trainer(
    #     max_epochs=config.max_epochs,
    #     logger=wandb_logger,
    #     callbacks=[checkpoint_callback, early_stop_callback],
    #     accelerator='auto',
    #     devices=1
    # )
    
    # Train model
    # trainer.fit(model, data_module)
    
    # Evaluate model
    # trainer.test(model, data_module)
    
    # Log final metrics
    # Placeholder: Random metrics for demonstration
    cell_wise_spearman = np.random.uniform(0.4, 0.8)
    gene_wise_spearman = np.random.uniform(0.1, 0.5)
    mse = np.random.uniform(0.1, 0.5)
    
    wandb.log({
        'final_cell_wise_spearman': cell_wise_spearman,
        'final_gene_wise_spearman': gene_wise_spearman,
        'final_mse': mse
    })
    
    # Return metrics for sweep to optimize
    return {
        'cell_wise_spearman': cell_wise_spearman,
        'gene_wise_spearman': gene_wise_spearman,
        'mse': mse
    }

def run_hyperparameter_search(config_path, sweep_config_path, count=10):
    """
    Run hyperparameter search using wandb sweeps.
    
    Parameters:
    - config_path: Path to main configuration file
    - sweep_config_path: Path to sweep configuration file
    - count: Number of runs to execute
    """
    # Load configurations
    config = load_config(config_path)
    sweep_config = load_config(sweep_config_path)
    
    # Add base config to sweep config
    sweep_config['parameters']['data_path'] = {'value': os.path.join(config['output_dir'], 'features', 'training_data.npz')}
    sweep_config['parameters']['output_dir'] = {'value': config['output_dir']}
    sweep_config['parameters']['wandb_project'] = {'value': config.get('wandb_project', 'spatial-transcriptomics')}
    sweep_config['parameters']['max_epochs'] = {'value': config.get('max_epochs', 100)}
    sweep_config['parameters']['patience'] = {'value': config.get('patience', 10)}
    
    # Initialize wandb sweep
    sweep_id = wandb.sweep(
        sweep_config,
        project=config.get('wandb_project', 'spatial-transcriptomics')
    )
    
    # Run sweep agent
    wandb.agent(sweep_id, function=train_model_with_config, count=count)
    
    # Save sweep results
    sweep_dir = os.path.join(config['output_dir'], 'sweeps')
    os.makedirs(sweep_dir, exist_ok=True)
    
    # Get best run from sweep
    api = wandb.Api()
    sweep = api.sweep(f"{config.get('wandb_project', 'spatial-transcriptomics')}/{sweep_id}")
    best_run = sweep.best_run()
    
    # Save best hyperparameters
    best_params = best_run.config
    best_metrics = {
        'cell_wise_spearman': best_run.summary.get('final_cell_wise_spearman', 0),
        'gene_wise_spearman': best_run.summary.get('final_gene_wise_spearman', 0),
        'mse': best_run.summary.get('final_mse', 0)
    }
    
    sweep_results = {
        'sweep_id': sweep_id,
        'best_run_id': best_run.id,
        'best_params': best_params,
        'best_metrics': best_metrics
    }
    
    with open(os.path.join(sweep_dir, 'sweep_results.yaml'), 'w') as f:
        yaml.dump(sweep_results, f)
    
    print(f"Hyperparameter search complete. Results saved to {os.path.join(sweep_dir, 'sweep_results.yaml')}")
    print(f"Best run: {best_run.id}")
    print(f"Best cell-wise Spearman: {best_metrics['cell_wise_spearman']:.4f}")
    print(f"Best gene-wise Spearman: {best_metrics['gene_wise_spearman']:.4f}")
    print(f"Best MSE: {best_metrics['mse']:.4f}")
    
    return sweep_results

if __name__ == "__main__":
    args = parse_args()
    run_hyperparameter_search(args.config, args.sweep_config, args.count)

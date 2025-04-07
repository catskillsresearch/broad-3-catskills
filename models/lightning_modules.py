import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, List, Tuple, Optional, Union
import wandb

class DeepSpotLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for training the unified DeepSpot model.
    
    This module handles:
    1. Training, validation, and testing loops
    2. Optimizer and learning rate scheduler configuration
    3. Metrics calculation and logging
    4. Model checkpointing
    """
    
    def __init__(self, model: nn.Module, config: Dict):
        """
        Initialize the lightning module.
        
        Args:
            model: The unified DeepSpot model
            config: Configuration dictionary with training parameters
        """
        super(DeepSpotLightningModule, self).__init__()
        self.model = model
        self.config = config
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 1e-4)
        self.weight_decay = config.get('weight_decay', 1e-5)
        self.lr_scheduler_factor = config.get('lr_scheduler_factor', 0.5)
        self.lr_scheduler_patience = config.get('lr_scheduler_patience', 10)
        
        # Loss function weights
        self.gene_wise_weight = config.get('gene_wise_weight', 0.3)
        self.cell_wise_weight = config.get('cell_wise_weight', 0.7)
        
        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['model'])
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            batch: Dictionary containing input tensors
            
        Returns:
            Dictionary containing model outputs
        """
        return self.model(
            spot_features=batch['spot_features'],
            subspot_features=batch['subspot_features'],
            neighbor_features=batch['neighbor_features'],
            subspot_distances=batch.get('subspot_distances'),
            neighbor_distances=batch.get('neighbor_distances'),
            spatial_coordinates=batch.get('spatial_coordinates')
        )
    
    def _calculate_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate loss for model outputs.
        
        Args:
            outputs: Dictionary containing model outputs
            targets: Dictionary containing target values
            
        Returns:
            Dictionary containing loss values
        """
        from unified_approach import balanced_spearman_loss, cell_wise_spearman_loss, gene_wise_spearman_loss
        
        # Calculate losses for measured genes
        measured_balanced_loss = balanced_spearman_loss(
            outputs['measured_predictions'],
            targets['measured_expressions'],
            gene_wise_weight=self.gene_wise_weight,
            cell_wise_weight=self.cell_wise_weight
        )
        
        measured_cell_wise_loss = cell_wise_spearman_loss(
            outputs['measured_predictions'],
            targets['measured_expressions']
        )
        
        measured_gene_wise_loss = gene_wise_spearman_loss(
            outputs['measured_predictions'],
            targets['measured_expressions']
        )
        
        # Calculate losses for unmeasured genes if targets are available
        if 'unmeasured_expressions' in targets:
            unmeasured_balanced_loss = balanced_spearman_loss(
                outputs['unmeasured_predictions'],
                targets['unmeasured_expressions'],
                gene_wise_weight=self.gene_wise_weight,
                cell_wise_weight=self.cell_wise_weight
            )
            
            unmeasured_cell_wise_loss = cell_wise_spearman_loss(
                outputs['unmeasured_predictions'],
                targets['unmeasured_expressions']
            )
            
            unmeasured_gene_wise_loss = gene_wise_spearman_loss(
                outputs['unmeasured_predictions'],
                targets['unmeasured_expressions']
            )
            
            # Calculate cell type classification loss if targets are available
            if 'cell_types' in targets and 'cell_type_logits' in outputs:
                cell_type_loss = F.cross_entropy(
                    outputs['cell_type_logits'],
                    targets['cell_types']
                )
            else:
                cell_type_loss = torch.tensor(0.0, device=self.device)
            
            # Combine losses
            total_loss = measured_balanced_loss + unmeasured_balanced_loss + 0.1 * cell_type_loss
        else:
            # If unmeasured gene targets are not available, use only measured gene loss
            unmeasured_balanced_loss = torch.tensor(0.0, device=self.device)
            unmeasured_cell_wise_loss = torch.tensor(0.0, device=self.device)
            unmeasured_gene_wise_loss = torch.tensor(0.0, device=self.device)
            cell_type_loss = torch.tensor(0.0, device=self.device)
            
            total_loss = measured_balanced_loss
        
        return {
            'total_loss': total_loss,
            'measured_balanced_loss': measured_balanced_loss,
            'measured_cell_wise_loss': measured_cell_wise_loss,
            'measured_gene_wise_loss': measured_gene_wise_loss,
            'unmeasured_balanced_loss': unmeasured_balanced_loss,
            'unmeasured_cell_wise_loss': unmeasured_cell_wise_loss,
            'unmeasured_gene_wise_loss': unmeasured_gene_wise_loss,
            'cell_type_loss': cell_type_loss
        }
    
    def _calculate_metrics(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate evaluation metrics for model outputs.
        
        Args:
            outputs: Dictionary containing model outputs
            targets: Dictionary containing target values
            
        Returns:
            Dictionary containing metric values
        """
        # Calculate Spearman correlation for measured genes
        measured_cell_wise_corr = self._spearman_correlation(
            outputs['measured_predictions'],
            targets['measured_expressions'],
            dim=1  # Cell-wise (across genes)
        )
        
        measured_gene_wise_corr = self._spearman_correlation(
            outputs['measured_predictions'],
            targets['measured_expressions'],
            dim=0  # Gene-wise (across cells)
        )
        
        # Calculate Spearman correlation for unmeasured genes if targets are available
        if 'unmeasured_expressions' in targets:
            unmeasured_cell_wise_corr = self._spearman_correlation(
                outputs['unmeasured_predictions'],
                targets['unmeasured_expressions'],
                dim=1  # Cell-wise (across genes)
            )
            
            unmeasured_gene_wise_corr = self._spearman_correlation(
                outputs['unmeasured_predictions'],
                targets['unmeasured_expressions'],
                dim=0  # Gene-wise (across cells)
            )
            
            # Calculate cell type classification accuracy if targets are available
            if 'cell_types' in targets and 'cell_type_logits' in outputs:
                cell_type_preds = torch.argmax(outputs['cell_type_logits'], dim=1)
                cell_type_acc = (cell_type_preds == targets['cell_types']).float().mean()
            else:
                cell_type_acc = torch.tensor(0.0, device=self.device)
        else:
            unmeasured_cell_wise_corr = torch.tensor(0.0, device=self.device)
            unmeasured_gene_wise_corr = torch.tensor(0.0, device=self.device)
            cell_type_acc = torch.tensor(0.0, device=self.device)
        
        return {
            'measured_cell_wise_corr': measured_cell_wise_corr,
            'measured_gene_wise_corr': measured_gene_wise_corr,
            'unmeasured_cell_wise_corr': unmeasured_cell_wise_corr,
            'unmeasured_gene_wise_corr': unmeasured_gene_wise_corr,
            'cell_type_acc': cell_type_acc
        }
    
    def _spearman_correlation(self, predictions: torch.Tensor, targets: torch.Tensor, dim: int = 0) -> torch.Tensor:
        """
        Calculate Spearman correlation coefficient.
        
        Args:
            predictions: Predicted values
            targets: Target values
            dim: Dimension along which to calculate correlation (0 for gene-wise, 1 for cell-wise)
            
        Returns:
            Mean Spearman correlation coefficient
        """
        # Convert to ranks
        def _to_ranks(x, dim):
            return torch.argsort(torch.argsort(x, dim=dim), dim=dim).float()
        
        # Handle different dimensions
        if dim == 0:  # Gene-wise (across cells)
            predictions = predictions.t()  # [n_genes, batch_size]
            targets = targets.t()  # [n_genes, batch_size]
            dim = 1  # Now calculate along dimension 1
        
        pred_ranks = _to_ranks(predictions, dim=dim)
        target_ranks = _to_ranks(targets, dim=dim)
        
        # Calculate mean rank
        pred_mean = pred_ranks.mean(dim=dim, keepdim=True)
        target_mean = target_ranks.mean(dim=dim, keepdim=True)
        
        # Calculate differences from mean
        pred_diff = pred_ranks - pred_mean
        target_diff = target_ranks - target_mean
        
        # Calculate covariance
        cov = (pred_diff * target_diff).sum(dim=dim)
        
        # Calculate standard deviations
        pred_std = torch.sqrt((pred_diff ** 2).sum(dim=dim) + 1e-8)
        target_std = torch.sqrt((target_diff ** 2).sum(dim=dim) + 1e-8)
        
        # Calculate correlation
        correlation = cov / (pred_std * target_std + 1e-8)
        
        # Return mean correlation
        return correlation.mean()
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Training step.
        
        Args:
            batch: Dictionary containing input tensors and targets
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing loss values
        """
        # Forward pass
        outputs = self(batch)
        
        # Calculate loss
        loss_dict = self._calculate_loss(outputs, batch)
        
        # Calculate metrics
        metrics_dict = self._calculate_metrics(outputs, batch)
        
        # Log metrics
        for name, value in {**loss_dict, **metrics_dict}.items():
            self.log(f'train/{name}', value, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss_dict
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step.
        
        Args:
            batch: Dictionary containing input tensors and targets
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing loss and metric values
        """
        # Forward pass
        outputs = self(batch)
        
        # Calculate loss
        loss_dict = self._calculate_loss(outputs, batch)
        
        # Calculate metrics
        metrics_dict = self._calculate_metrics(outputs, batch)
        
        # Log metrics
        for name, value in {**loss_dict, **metrics_dict}.items():
            self.log(f'val/{name}', value, on_step=False, on_epoch=True, prog_bar=True)
        
        return {**loss_dict, **metrics_dict}
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Test step.
        
        Args:
            batch: Dictionary containing input tensors and targets
            batch_idx: Index of the current batch
            
        Returns:
            Dictionary containing loss and metric values
        """
        # Forward pass
        outputs = self(batch)
        
        # Calculate loss
        loss_dict = self._calculate_loss(outputs, batch)
        
        # Calculate metrics
        metrics_dict = self._calculate_metrics(outputs, batch)
        
        # Log metrics
        for name, value in {**loss_dict, **metrics_dict}.items():
            self.log(f'test/{name}', value, on_step=False, on_epoch=True)
        
        return {**loss_dict, **metrics_dict}
    
    def configure_optimizers(self) -> Dict:
        """
        Configure optimizer and learning rate scheduler.
        
        Returns:
            Dictionary containing optimizer and scheduler configuration
        """
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Create learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.lr_scheduler_factor,
            patience=self.lr_scheduler_patience,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val/total_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

class DeepSpotDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for the DeepSpot dataset.
    
    This module handles:
    1. Data loading and preprocessing
    2. Dataset splitting (train, validation, test)
    3. Batch creation and data augmentation
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the data module.
        
        Args:
            config: Configuration dictionary with dataset parameters
        """
        super(DeepSpotDataModule, self).__init__()
        self.config = config
        
        # Dataset parameters
        self.data_dir = config.get('data_dir', 'data')
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 4)
        self.val_split = config.get('val_split', 0.1)
        self.test_split = config.get('test_split', 0.1)
        self.use_synthetic = config.get('use_synthetic', False)
        self.use_small_dataset = config.get('use_small_dataset', False)
        
        # Dataset objects
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """
        Download or prepare the dataset.
        This method is called only once and on the main process.
        """
        # This method is used for downloading or preparing data
        # In our case, we assume the data is already prepared
        pass
    
    def setup(self, stage: Optional[str] = None):
        """
        Set up the datasets for the given stage.
        
        Args:
            stage: Current stage ('fit', 'validate', 'test', or None)
        """
        import os
        import numpy as np
        from torch.utils.data import Dataset, random_split
        
        # Define dataset class
        class DeepSpotDataset(Dataset):
            def __init__(self, data_dir, split='train', use_synthetic=False, use_small=False):
                self.data_dir = data_dir
                self.split = split
                
                # Determine dataset path based on configuration
                if use_synthetic:
                    dataset_type = 'synthetic'
                elif use_small:
                    dataset_type = 'small'
                else:
                    dataset_type = 'full'
                
                # Load data
                data_path = os.path.join(data_dir, f'{dataset_type}_{split}.npz')
                data = np.load(data_path, allow_pickle=True)
                
                # Extract data
                self.spot_features = data['spot_features']
                self.subspot_features = data['subspot_features']
                self.neighbor_features = data['neighbor_features']
                self.measured_expressions = data['measured_expressions']
                
                # Optional data
                self.subspot_distances = data['subspot_distances'] if 'subspot_distances' in data else None
                self.neighbor_distances = data['neighbor_distances'] if 'neighbor_distances' in data else None
                self.spatial_coordinates = data['spatial_coordinates'] if 'spatial_coordinates' in data else None
                self.unmeasured_expressions = data['unmeasured_expressions'] if 'unmeasured_expressions' in data else None
                self.cell_types = data['cell_types'] if 'cell_types' in data else None
                self.cell_labels = data['cell_labels'] if 'cell_labels' in data else None
            
            def __len__(self):
                return len(self.spot_features)
            
            def __getitem__(self, idx):
                item = {
                    'spot_features': torch.tensor(self.spot_features[idx], dtype=torch.float32),
                    'subspot_features': torch.tensor(self.subspot_features[idx], dtype=torch.float32),
                    'neighbor_features': torch.tensor(self.neighbor_features[idx], dtype=torch.float32),
                    'measured_expressions': torch.tensor(self.measured_expressions[idx], dtype=torch.float32)
                }
                
                # Add optional data if available
                if self.subspot_distances is not None:
                    item['subspot_distances'] = torch.tensor(self.subspot_distances[idx], dtype=torch.float32)
                
                if self.neighbor_distances is not None:
                    item['neighbor_distances'] = torch.tensor(self.neighbor_distances[idx], dtype=torch.float32)
                
                if self.spatial_coordinates is not None:
                    item['spatial_coordinates'] = torch.tensor(self.spatial_coordinates[idx], dtype=torch.float32)
                
                if self.unmeasured_expressions is not None:
                    item['unmeasured_expressions'] = torch.tensor(self.unmeasured_expressions[idx], dtype=torch.float32)
                
                if self.cell_types is not None:
                    item['cell_types'] = torch.tensor(self.cell_types[idx], dtype=torch.long)
                
                if self.cell_labels is not None:
                    item['cell_labels'] = self.cell_labels[idx]
                
                return item
        
        # Set up datasets based on stage
        if stage == 'fit' or stage is None:
            # Create full dataset
            full_dataset = DeepSpotDataset(
                data_dir=self.data_dir,
                split='train',
                use_synthetic=self.use_synthetic,
                use_small=self.use_small_dataset
            )
            
            # Split into train and validation sets
            val_size = int(len(full_dataset) * self.val_split)
            train_size = len(full_dataset) - val_size
            
            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
        
        if stage == 'test' or stage is None:
            # Create test dataset
            self.test_dataset = DeepSpotDataset(
                data_dir=self.data_dir,
                split='test',
                use_synthetic=self.use_synthetic,
                use_small=self.use_small_dataset
            )
    
    def train_dataloader(self):
        """
        Create the training data loader.
        
        Returns:
            Training data loader
        """
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """
        Create the validation data loader.
        
        Returns:
            Validation data loader
        """
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """
        Create the test data loader.
        
        Returns:
            Test data loader
        """
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

def train_model(config: Dict):
    """
    Train the unified DeepSpot model.
    
    Args:
        config: Configuration dictionary with training parameters
    """
    import os
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import WandbLogger
    from unified_approach import UnifiedDeepSpotModel
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config.get('wandb_project', 'deepspot'),
        name=config.get('wandb_run_name', 'unified_model'),
        log_model=True
    )
    
    # Initialize model
    model = UnifiedDeepSpotModel(config)
    
    # Initialize lightning module
    lightning_module = DeepSpotLightningModule(model, config)
    
    # Initialize data module
    data_module = DeepSpotDataModule(config)
    
    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config.get('output_dir', 'output'), 'checkpoints'),
        filename='deepspot-{epoch:02d}-{val/measured_cell_wise_corr:.4f}',
        monitor='val/measured_cell_wise_corr',
        mode='max',
        save_top_k=3,
        save_last=True
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val/measured_cell_wise_corr',
        mode='max',
        patience=config.get('early_stopping_patience', 20),
        verbose=True
    )
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=config.get('max_epochs', 100),
        gpus=1 if torch.cuda.is_available() else 0,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        gradient_clip_val=config.get('gradient_clip_val', 1.0),
        log_every_n_steps=config.get('log_every_n_steps', 10),
        deterministic=True
    )
    
    # Train model
    trainer.fit(lightning_module, data_module)
    
    # Test model
    trainer.test(lightning_module, data_module)
    
    # Return best checkpoint path
    return checkpoint_callback.best_model_path

def evaluate_model(config: Dict, checkpoint_path: str):
    """
    Evaluate the trained model.
    
    Args:
        config: Configuration dictionary with evaluation parameters
        checkpoint_path: Path to model checkpoint
    """
    import os
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import WandbLogger
    from unified_approach import UnifiedDeepSpotModel
    
    # Initialize wandb logger
    wandb_logger = WandbLogger(
        project=config.get('wandb_project', 'deepspot'),
        name=config.get('wandb_run_name', 'unified_model_eval'),
        log_model=False
    )
    
    # Initialize model
    model = UnifiedDeepSpotModel(config)
    
    # Initialize lightning module
    lightning_module = DeepSpotLightningModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        config=config
    )
    
    # Initialize data module
    data_module = DeepSpotDataModule(config)
    
    # Initialize trainer
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        logger=wandb_logger,
        deterministic=True
    )
    
    # Test model
    results = trainer.test(lightning_module, data_module)
    
    # Save results
    import json
    output_dir = config.get('output_dir', 'output')
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results[0], f, indent=4)
    
    return results[0]

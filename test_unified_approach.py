#!/usr/bin/env python3
# test_unified_approach.py

"""
This script tests the unified approach that combines the sophisticated neural network
architecture from yesterday with the comprehensive framework from today.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import local modules
from broad_3_catskills.models.unified_approach import (
    UnifiedDeepSpotModel,
    UnifiedPipeline,
    UnifiedLightningModule,
    spearman_correlation_loss,
    cell_wise_spearman_loss
)

def create_dummy_data():
    """Create dummy data for testing."""
    # Create dummy config
    config = {
        'feature_dim': 512,
        'phi_dim': 256,
        'embedding_dim': 512,
        'n_heads': 4,
        'n_cell_types': 10,
        'n_measured_genes': 460,
        'n_unmeasured_genes': 1000,
        'output_dir': '/tmp/unified_approach_test',
        'gene_wise_weight': 0.3,
        'cell_wise_weight': 0.7,
        'learning_rate': 0.001,
        'weight_decay': 0.0001,
        'dropout': 0.3
    }
    
    # Create dummy data
    n_spots = 100
    n_subspots = 5
    n_neighbors = 10
    
    spot_features = np.random.randn(n_spots, config['feature_dim'])
    subspot_features = np.random.randn(n_spots, n_subspots, config['feature_dim'])
    neighbor_features = np.random.randn(n_spots, n_neighbors, config['feature_dim'])
    neighbor_distances = np.random.rand(n_spots, n_neighbors)
    
    measured_gene_expression = np.random.randn(n_spots, config['n_measured_genes'])
    
    # Create dummy cell labels (50% dysplastic, 50% non-dysplastic)
    cell_labels = np.zeros(n_spots)
    cell_labels[:n_spots//2] = 1
    
    # Create dummy gene names
    measured_gene_names = [f"measured_gene_{i}" for i in range(config['n_measured_genes'])]
    unmeasured_gene_names = [f"unmeasured_gene_{i}" for i in range(config['n_unmeasured_genes'])]
    
    # Create dummy reference data
    reference_data = {
        'measured': np.random.randn(config['n_cell_types'], config['n_measured_genes']),
        'unmeasured': np.random.randn(config['n_cell_types'], config['n_unmeasured_genes'])
    }
    
    return {
        'config': config,
        'spot_features': spot_features,
        'subspot_features': subspot_features,
        'neighbor_features': neighbor_features,
        'neighbor_distances': neighbor_distances,
        'measured_gene_expression': measured_gene_expression,
        'cell_labels': cell_labels,
        'measured_gene_names': measured_gene_names,
        'unmeasured_gene_names': unmeasured_gene_names,
        'reference_data': reference_data
    }

def test_unified_model():
    """Test the unified model."""
    print("\n=== Testing Unified Model ===")
    
    # Create dummy data
    dummy_data = create_dummy_data()
    config = dummy_data['config']
    
    # Create model
    model = UnifiedDeepSpotModel(config)
    
    # Convert data to tensors
    spot_features = torch.tensor(dummy_data['spot_features'], dtype=torch.float32)
    subspot_features = torch.tensor(dummy_data['subspot_features'], dtype=torch.float32)
    neighbor_features = torch.tensor(dummy_data['neighbor_features'], dtype=torch.float32)
    neighbor_distances = torch.tensor(dummy_data['neighbor_distances'], dtype=torch.float32)
    
    # Set reference data
    reference_measured = torch.tensor(dummy_data['reference_data']['measured'], dtype=torch.float32)
    reference_unmeasured = torch.tensor(dummy_data['reference_data']['unmeasured'], dtype=torch.float32)
    model.set_reference_data(reference_measured, reference_unmeasured)
    
    # Forward pass
    with torch.no_grad():
        predictions = model(
            spot_features,
            subspot_features,
            neighbor_features,
            neighbor_distances,
            predict_unmeasured=True
        )
    
    # Check output shapes
    measured_shape = tuple(predictions['measured_predictions'].shape)
    expected_measured_shape = (dummy_data['spot_features'].shape[0], config['n_measured_genes'])
    
    unmeasured_shape = tuple(predictions['unmeasured_predictions'].shape)
    expected_unmeasured_shape = (dummy_data['spot_features'].shape[0], config['n_unmeasured_genes'])
    
    print(f"Measured predictions shape: {measured_shape}, Expected: {expected_measured_shape}")
    print(f"Unmeasured predictions shape: {unmeasured_shape}, Expected: {expected_unmeasured_shape}")
    
    # Check if cell embedding and cell type logits are present
    has_cell_embedding = 'cell_embedding' in predictions
    has_cell_type_logits = 'cell_type_logits' in predictions
    
    print(f"Has cell embedding: {has_cell_embedding}")
    print(f"Has cell type logits: {has_cell_type_logits}")
    
    # Check shapes match
    shapes_match = (measured_shape == expected_measured_shape and 
                   unmeasured_shape == expected_unmeasured_shape)
    
    print(f"Test passed: {shapes_match and has_cell_embedding and has_cell_type_logits}")
    
    return shapes_match and has_cell_embedding and has_cell_type_logits

def test_loss_functions():
    """Test the loss functions."""
    print("\n=== Testing Loss Functions ===")
    
    # Create dummy predictions and targets
    batch_size = 10
    n_genes = 100
    
    predictions = torch.randn(batch_size, n_genes)
    targets = torch.randn(batch_size, n_genes)
    
    # Compute losses
    gene_wise_loss = spearman_correlation_loss(predictions, targets)
    cell_wise_loss = cell_wise_spearman_loss(predictions, targets)
    
    # Check loss values
    print(f"Gene-wise Spearman loss: {gene_wise_loss.item():.4f}")
    print(f"Cell-wise Spearman loss: {cell_wise_loss.item():.4f}")
    
    # Check if loss values are valid
    valid_gene_wise = 0 <= gene_wise_loss.item() <= 2.0  # Should be around 1.0 for random data
    valid_cell_wise = 0 <= cell_wise_loss.item() <= 2.0  # Should be around 1.0 for random data
    
    print(f"Valid gene-wise loss: {valid_gene_wise}")
    print(f"Valid cell-wise loss: {valid_cell_wise}")
    
    # Test passed if both losses are valid
    test_passed = valid_gene_wise and valid_cell_wise
    print(f"Test passed: {test_passed}")
    
    return test_passed

def test_lightning_module():
    """Test the PyTorch Lightning module."""
    print("\n=== Testing Lightning Module ===")
    
    # Create dummy data
    dummy_data = create_dummy_data()
    config = dummy_data['config']
    
    # Create model
    model = UnifiedLightningModule(config)
    
    # Create dummy batch
    batch = {
        'spot_features': torch.tensor(dummy_data['spot_features'], dtype=torch.float32),
        'subspot_features': torch.tensor(dummy_data['subspot_features'], dtype=torch.float32),
        'neighbor_features': torch.tensor(dummy_data['neighbor_features'], dtype=torch.float32),
        'neighbor_distances': torch.tensor(dummy_data['neighbor_distances'], dtype=torch.float32),
        'measured_expression': torch.tensor(dummy_data['measured_gene_expression'], dtype=torch.float32)
    }
    
    # Test forward pass
    predictions = model(
        batch['spot_features'],
        batch['subspot_features'],
        batch['neighbor_features'],
        batch['neighbor_distances']
    )
    
    # Check if predictions contain measured predictions
    has_measured = 'measured_predictions' in predictions
    
    # Test training step
    try:
        loss = model.training_step(batch, 0)
        training_step_works = True
        print(f"Training step loss: {loss.item():.4f}")
    except Exception as e:
        training_step_works = False
        print(f"Training step error: {str(e)}")
    
    # Test validation step
    try:
        model.validation_step(batch, 0)
        validation_step_works = True
        print("Validation step works")
    except Exception as e:
        validation_step_works = False
        print(f"Validation step error: {str(e)}")
    
    # Test test step
    try:
        model.test_step(batch, 0)
        test_step_works = True
        print("Test step works")
    except Exception as e:
        test_step_works = False
        print(f"Test step error: {str(e)}")
    
    # Test predict step
    try:
        pred = model.predict_step(batch, 0)
        predict_step_works = True
        print("Predict step works")
    except Exception as e:
        predict_step_works = False
        print(f"Predict step error: {str(e)}")
    
    # Test configure optimizers
    try:
        optim_config = model.configure_optimizers()
        has_optimizer = 'optimizer' in optim_config
        has_scheduler = 'lr_scheduler' in optim_config
        configure_optimizers_works = has_optimizer and has_scheduler
        print(f"Configure optimizers works: {configure_optimizers_works}")
    except Exception as e:
        configure_optimizers_works = False
        print(f"Configure optimizers error: {str(e)}")
    
    # Test passed if all components work
    test_passed = (has_measured and training_step_works and validation_step_works and 
                  test_step_works and predict_step_works and configure_optimizers_works)
    
    print(f"Test passed: {test_passed}")
    
    return test_passed

def test_unified_pipeline():
    """Test the unified pipeline."""
    print("\n=== Testing Unified Pipeline ===")
    
    # Create dummy data
    dummy_data = create_dummy_data()
    config = dummy_data['config']
    
    # Create pipeline
    pipeline = UnifiedPipeline(config)
    
    # Test run_crunch1_and_2
    try:
        predictions = pipeline.run_crunch1_and_2(
            dummy_data['spot_features'],
            dummy_data['subspot_features'],
            dummy_data['neighbor_features'],
            dummy_data['neighbor_distances'],
            dummy_data['measured_gene_expression'],
            dummy_data['reference_data']
        )
        
        has_measured = 'measured_predictions' in predictions
        has_unmeasured = 'unmeasured_predictions' in predictions
        
        measured_shape = predictions['measured_predictions'].shape
        expected_measured_shape = (dummy_data['spot_features'].shape[0], config['n_measured_genes'])
        
        unmeasured_shape = predictions['unmeasured_predictions'].shape
        expected_unmeasured_shape = (dummy_data['spot_features'].shape[0], config['n_unmeasured_genes'])
        
        crunch1_and_2_works = (has_measured and has_unmeasured and 
                              measured_shape == expected_measured_shape and 
                              unmeasured_shape == expected_unmeasured_shape)
        
        print(f"Crunch 1 & 2 works: {crunch1_and_2_works}")
        print(f"Measured shape: {measured_shape}, Expected: {expected_measured_shape}")
        print(f"Unmeasured shape: {unmeasured_shape}, Expected: {expected_unmeasured_shape}")
    except Exception as e:
        crunch1_and_2_works = False
        print(f"Crunch 1 & 2 error: {str(e)}")
    
    # Test run_crunch3
    try:
        # Combine measured and unmeasured gene expression
        all_gene_expression = np.concatenate([
            dummy_data['measured_gene_expression'],
            np.random.randn(dummy_data['spot_features'].shape[0], config['n_unmeasured_genes'])
        ], axis=1)
        
        all_gene_names = dummy_data['measured_gene_names'] + dummy_data['unmeasured_gene_names']
        
        rankings = pipeline.run_crunch3(
            all_gene_expression,
            dummy_data['cell_labels'],
            all_gene_names
        )
        
        has_rankings = rankings is not None
        has_expected_columns = all(col in rankings.columns for col in ['gene_name', 'logFC', 'abs_logFC', 'rank'])
        has_expected_rows = len(rankings) == len(all_gene_names)
        
        crunch3_works = has_rankings and has_expected_columns and has_expected_rows
        
        print(f"Crunch 3 works: {crunch3_works}")
        print(f"Rankings shape: {rankings.shape}")
        print(f"Has expected columns: {has_expected_columns}")
        print(f"Has expected rows: {has_expected_rows}")
    except Exception as e:
        crunch3_works = False
        print(f"Crunch 3 error: {str(e)}")
    
    # Test run_full_pipeline
    try:
        results = pipeline.run_full_pipeline(
            dummy_data['spot_features'],
            dummy_data['subspot_features'],
            dummy_data['neighbor_features'],
            dummy_data['neighbor_distances'],
            dummy_data['measured_gene_expression'],
            dummy_data['cell_labels'],
            dummy_data['measured_gene_names'],
            dummy_data['unmeasured_gene_names'],
            dummy_data['reference_data']
        )
        
        expected_keys = ['measured_predictions', 'unmeasured_predictions', 'combined_expression', 'gene_rankings']
        has_all_keys = all(key in results for key in expected_keys)
        
        full_pipeline_works = has_all_keys
        
        print(f"Full pipeline works: {full_pipeline_works}")
        print(f"Results keys: {list(results.keys())}")
    except Exception as e:
        full_pipeline_works = False
        print(f"Full pipeline error: {str(e)}")
    
    # Test passed if all components work
    test_passed = crunch1_and_2_works and crunch3_works and full_pipeline_works
    
    print(f"Test passed: {test_passed}")
    
    return test_passed

if __name__ == "__main__":
    print("Testing unified approach...")
    
    # Test individual components
    model_test_passed = test_unified_model()
    loss_test_passed = test_loss_functions()
    lightning_test_passed = test_lightning_module()
    pipeline_test_passed = test_unified_pipeline()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Unified Model: {'PASSED' if model_test_passed else 'FAILED'}")
    print(f"Loss Functions: {'PASSED' if loss_test_passed else 'FAILED'}")
    print(f"Lightning Module: {'PASSED' if lightning_test_passed else 'FAILED'}")
    print(f"Unified Pipeline: {'PASSED' if pipeline_test_passed else 'FAILED'}")
    
    # Overall result
    all_passed = model_test_passed and loss_test_passed and lightning_test_passed and pipeline_test_passed
    print(f"\nOverall Test Result: {'PASSED' if all_passed else 'FAILED'}")

#!/usr/bin/env python3
# test_integrated_pipeline.py

"""
This script tests the integrated pipeline that combines DeepSpot (Crunch 1),
Tarandros (Crunch 2), and logFC (Crunch 3) approaches.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Import local modules
from broad_3_catskills.models.integrated_pipeline import IntegratedPipeline
from broad_3_catskills.models.deepspot_model import DeepSpotModel
from broad_3_catskills.models.tarandros_model import TarandrosModel
from broad_3_catskills.models.logfc_gene_ranking import LogFCGeneRanking

def create_dummy_data():
    """Create dummy data for testing."""
    # Create dummy config
    config = {
        'feature_dim': 512,
        'embedding_dim': 256,
        'n_cell_types': 10,
        'n_measured_genes': 460,
        'n_unmeasured_genes': 1000,
        'output_dir': '/tmp/integrated_pipeline_test'
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

def test_deepspot_model():
    """Test DeepSpot model."""
    print("\n=== Testing DeepSpot Model (Crunch 1) ===")
    
    # Create dummy data
    dummy_data = create_dummy_data()
    config = dummy_data['config']
    
    # Create model
    model = DeepSpotModel(config)
    
    # Convert data to tensors
    spot_features = torch.tensor(dummy_data['spot_features'], dtype=torch.float32)
    subspot_features = torch.tensor(dummy_data['subspot_features'], dtype=torch.float32)
    neighbor_features = torch.tensor(dummy_data['neighbor_features'], dtype=torch.float32)
    neighbor_distances = torch.tensor(dummy_data['neighbor_distances'], dtype=torch.float32)
    
    # Forward pass
    with torch.no_grad():
        output = model(spot_features, subspot_features, neighbor_features, neighbor_distances)
    
    # Check output shape
    expected_shape = (dummy_data['spot_features'].shape[0], config['n_measured_genes'])
    actual_shape = tuple(output.shape)
    
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape: {actual_shape}")
    print(f"Test passed: {expected_shape == actual_shape}")
    
    return expected_shape == actual_shape

def test_tarandros_model():
    """Test Tarandros model."""
    print("\n=== Testing Tarandros Model (Crunch 2) ===")
    
    # Create dummy data
    dummy_data = create_dummy_data()
    config = dummy_data['config']
    
    # Create model
    model = TarandrosModel(config)
    
    # Convert data to tensors
    spot_features = torch.tensor(dummy_data['spot_features'], dtype=torch.float32)
    measured_expressions = torch.tensor(dummy_data['measured_gene_expression'], dtype=torch.float32)
    neighbor_features = torch.tensor(dummy_data['neighbor_features'], dtype=torch.float32)
    neighbor_distances = torch.tensor(dummy_data['neighbor_distances'], dtype=torch.float32)
    
    # Set reference data
    reference_measured = torch.tensor(dummy_data['reference_data']['measured'], dtype=torch.float32)
    reference_unmeasured = torch.tensor(dummy_data['reference_data']['unmeasured'], dtype=torch.float32)
    model.set_reference_data(reference_measured, reference_unmeasured)
    
    # Forward pass
    with torch.no_grad():
        output = model(spot_features, measured_expressions, neighbor_features, neighbor_distances)
    
    # Check output shape
    expected_shape = (dummy_data['spot_features'].shape[0], config['n_unmeasured_genes'])
    actual_shape = tuple(output.shape)
    
    print(f"Expected output shape: {expected_shape}")
    print(f"Actual output shape: {actual_shape}")
    print(f"Test passed: {expected_shape == actual_shape}")
    
    return expected_shape == actual_shape

def test_logfc_ranking():
    """Test logFC gene ranking."""
    print("\n=== Testing LogFC Gene Ranking (Crunch 3) ===")
    
    # Create dummy data
    dummy_data = create_dummy_data()
    config = dummy_data['config']
    
    # Create ranker
    ranker = LogFCGeneRanking(config)
    
    # Create dummy gene expression data
    n_spots = dummy_data['spot_features'].shape[0]
    n_genes = 500
    gene_expression = np.random.randn(n_spots, n_genes)
    gene_names = [f"gene_{i}" for i in range(n_genes)]
    
    # Split by cell labels
    dysplastic_mask = dummy_data['cell_labels'] == 1
    non_dysplastic_mask = dummy_data['cell_labels'] == 0
    
    dysplastic_expression = gene_expression[dysplastic_mask]
    non_dysplastic_expression = gene_expression[non_dysplastic_mask]
    
    # Rank genes
    rankings = ranker.rank_genes(dysplastic_expression, non_dysplastic_expression, gene_names)
    
    # Check rankings shape
    expected_shape = (n_genes, 4)  # gene_name, logFC, abs_logFC, rank
    actual_shape = rankings.shape
    
    print(f"Expected rankings shape: {expected_shape}")
    print(f"Actual rankings shape: {actual_shape}")
    print(f"Test passed: {expected_shape == actual_shape}")
    
    # Check if rankings are sorted by abs_logFC
    is_sorted = all(rankings['abs_logFC'].iloc[i] >= rankings['abs_logFC'].iloc[i+1] for i in range(len(rankings)-1))
    print(f"Rankings sorted by abs_logFC: {is_sorted}")
    
    return expected_shape == actual_shape and is_sorted

def test_integrated_pipeline():
    """Test integrated pipeline."""
    print("\n=== Testing Integrated Pipeline ===")
    
    # Create dummy data
    dummy_data = create_dummy_data()
    config = dummy_data['config']
    
    # Create pipeline
    pipeline = IntegratedPipeline(config)
    
    # Run full pipeline
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
        
        # Check results
        expected_keys = ['crunch1_predictions', 'crunch2_predictions', 'combined_expression', 'gene_rankings']
        actual_keys = list(results.keys())
        
        print(f"Expected result keys: {expected_keys}")
        print(f"Actual result keys: {actual_keys}")
        print(f"Test passed: {all(k in actual_keys for k in expected_keys)}")
        
        # Check shapes
        crunch1_shape = results['crunch1_predictions'].shape
        expected_crunch1_shape = (dummy_data['spot_features'].shape[0], config['n_measured_genes'])
        
        crunch2_shape = results['crunch2_predictions'].shape
        expected_crunch2_shape = (dummy_data['spot_features'].shape[0], config['n_unmeasured_genes'])
        
        combined_shape = results['combined_expression'].shape
        expected_combined_shape = (dummy_data['spot_features'].shape[0], 
                                  config['n_measured_genes'] + config['n_unmeasured_genes'])
        
        print(f"Crunch1 shape: {crunch1_shape}, Expected: {expected_crunch1_shape}")
        print(f"Crunch2 shape: {crunch2_shape}, Expected: {expected_crunch2_shape}")
        print(f"Combined shape: {combined_shape}, Expected: {expected_combined_shape}")
        
        shapes_match = (crunch1_shape == expected_crunch1_shape and 
                        crunch2_shape == expected_crunch2_shape and 
                        combined_shape == expected_combined_shape)
        
        print(f"Shapes match: {shapes_match}")
        
        return all(k in actual_keys for k in expected_keys) and shapes_match
    
    except Exception as e:
        print(f"Error testing integrated pipeline: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing integrated pipeline components...")
    
    # Test individual components
    deepspot_test_passed = test_deepspot_model()
    tarandros_test_passed = test_tarandros_model()
    logfc_test_passed = test_logfc_ranking()
    
    # Test integrated pipeline
    pipeline_test_passed = test_integrated_pipeline()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"DeepSpot Model (Crunch 1): {'PASSED' if deepspot_test_passed else 'FAILED'}")
    print(f"Tarandros Model (Crunch 2): {'PASSED' if tarandros_test_passed else 'FAILED'}")
    print(f"LogFC Ranking (Crunch 3): {'PASSED' if logfc_test_passed else 'FAILED'}")
    print(f"Integrated Pipeline: {'PASSED' if pipeline_test_passed else 'FAILED'}")
    
    # Overall result
    all_passed = deepspot_test_passed and tarandros_test_passed and logfc_test_passed and pipeline_test_passed
    print(f"\nOverall Test Result: {'PASSED' if all_passed else 'FAILED'}")

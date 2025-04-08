#!/usr/bin/env python3
# test_visualization_implementation.py

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization module
from visualization_module import VisualizationModule

def main():
    """Test the visualization module implementation."""
    print("Testing visualization module implementation...")
    
    # Create output directory
    output_dir = os.path.join(os.getcwd(), 'test_visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualization module
    vis_module = VisualizationModule(
        output_dir=output_dir,
        experiment_name='test_visualization',
        use_wandb=False  # Disable wandb for testing
    )
    
    # Generate synthetic data for testing
    n_cells = 100
    n_genes = 50
    
    # Generate random gene expression data
    gene_expression = np.random.rand(n_cells, n_genes)
    
    # Generate random cell coordinates
    cell_coordinates = np.random.rand(n_cells, 2) * 100
    
    # Generate random region labels
    region_labels = np.random.randint(0, 3, n_cells)
    
    # Generate random quality scores
    quality_scores = np.random.rand(n_cells)
    
    # Generate random gene names
    gene_names = [f'Gene_{i}' for i in range(n_genes)]
    
    # Generate random predictions
    predictions = gene_expression + np.random.normal(0, 0.2, (n_cells, n_genes))
    
    print("\n1. Testing dataset overview visualization...")
    try:
        dataset_overview_path = vis_module.create_dataset_overview(
            gene_expression=gene_expression,
            cell_coordinates=cell_coordinates,
            region_labels=region_labels,
            quality_scores=quality_scores,
            gene_names=gene_names
        )
        print(f"  ✓ Dataset overview visualization created: {dataset_overview_path}")
    except Exception as e:
        print(f"  ✗ Failed to create dataset overview visualization: {e}")
    
    print("\n2. Testing spatial expression map visualization...")
    try:
        spatial_expression_path = vis_module.create_spatial_expression_map(
            cell_coordinates=cell_coordinates,
            expression_values=gene_expression,
            gene_names=gene_names
        )
        print(f"  ✓ Spatial expression map visualization created: {spatial_expression_path}")
    except Exception as e:
        print(f"  ✗ Failed to create spatial expression map visualization: {e}")
    
    print("\n3. Testing prediction accuracy visualization...")
    try:
        prediction_accuracy_path = vis_module.create_prediction_accuracy_visualization(
            predictions=predictions,
            targets=gene_expression,
            gene_names=gene_names
        )
        print(f"  ✓ Prediction accuracy visualization created: {prediction_accuracy_path}")
    except Exception as e:
        print(f"  ✗ Failed to create prediction accuracy visualization: {e}")
    
    print("\n4. Testing dimensionality reduction visualizations...")
    try:
        vis_paths = vis_module.create_dimensionality_reduction_visualizations(
            predictions=predictions,
            targets=gene_expression,
            vis_dir=os.path.join(output_dir, 'visualizations')
        )
        print(f"  ✓ Dimensionality reduction visualizations created:")
        for key, path in vis_paths.items():
            print(f"    - {key}: {path}")
    except Exception as e:
        print(f"  ✗ Failed to create dimensionality reduction visualizations: {e}")
    
    print("\n5. Testing volcano plot visualization...")
    try:
        volcano_plot_path = vis_module.create_volcano_plot(
            expression_data=gene_expression,
            region_labels=region_labels,
            gene_names=gene_names
        )
        print(f"  ✓ Volcano plot visualization created: {volcano_plot_path}")
    except Exception as e:
        print(f"  ✗ Failed to create volcano plot visualization: {e}")
    
    print("\nAll visualization tests completed.")
    print(f"Visualization outputs saved to: {output_dir}")

if __name__ == "__main__":
    main()

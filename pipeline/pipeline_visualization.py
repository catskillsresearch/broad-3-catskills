#!/usr/bin/env python3
# pipeline/pipeline_visualization.py

import luigi
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import visualization module
from visualization_module import VisualizationModule

class VisualizeDataset(luigi.Task):
    """Luigi task for creating dataset visualizations."""
    config_path = luigi.Parameter(description="Path to the configuration file")
    
    def requires(self):
        from pipeline.pipeline_data_preparation import PrepareTrainingData
        return PrepareTrainingData(config_path=self.config_path)
    
    def output(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        vis_dir = os.path.join(output_dir, 'visualizations')
        return luigi.LocalTarget(os.path.join(vis_dir, 'dataset_overview.png'))
    
    def run(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        # Load training data
        data = np.load(self.requires().output().path, allow_pickle=True)
        
        # Extract features and labels
        spot_features = data['spot_features']
        subspot_features = data['subspot_features']
        neighbor_features = data['neighbor_features']
        gene_expression = data['gene_expression']
        
        # Extract cell coordinates and region labels if available
        cell_coordinates = data.get('cell_coordinates', None)
        region_labels = data.get('region_labels', None)
        quality_scores = data.get('quality_scores', None)
        gene_names = data.get('gene_names', None)
        
        # If cell coordinates are not available, create dummy coordinates
        if cell_coordinates is None:
            print("Cell coordinates not found, creating dummy coordinates...")
            n_cells = gene_expression.shape[0]
            cell_coordinates = np.random.rand(n_cells, 2) * 100
        
        # If region labels are not available, create dummy labels
        if region_labels is None:
            print("Region labels not found, creating dummy labels...")
            n_cells = gene_expression.shape[0]
            region_labels = np.zeros(n_cells, dtype=int)
        
        # Initialize visualization module
        vis_module = VisualizationModule(
            output_dir=output_dir,
            experiment_name=f"dataset_visualization_{os.path.basename(self.config_path)}",
            use_wandb=True
        )
        
        # Create dataset overview visualization
        vis_module.create_dataset_overview(
            gene_expression=gene_expression,
            cell_coordinates=cell_coordinates,
            region_labels=region_labels,
            quality_scores=quality_scores,
            gene_names=gene_names
        )
        
        # Create spatial expression map visualization
        vis_module.create_spatial_expression_map(
            cell_coordinates=cell_coordinates,
            expression_values=gene_expression,
            gene_names=gene_names
        )
        
        print(f"Dataset visualizations complete. Saved to {os.path.dirname(self.output().path)}")


class VisualizeModelPredictions(luigi.Task):
    """Luigi task for creating model prediction visualizations."""
    config_path = luigi.Parameter(description="Path to the configuration file")
    experiment_name = luigi.Parameter(default='default_experiment', description="Name of the experiment")
    
    def requires(self):
        from pipeline.pipeline_model_training import PredictGeneExpression, PrepareTrainingData
        return {
            'predictions': PredictGeneExpression(config_path=self.config_path, experiment_name=self.experiment_name),
            'data': PrepareTrainingData(config_path=self.config_path)
        }
    
    def output(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        vis_dir = os.path.join(output_dir, 'visualizations', self.experiment_name)
        return luigi.LocalTarget(os.path.join(vis_dir, 'prediction_accuracy.png'))
    
    def run(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        vis_dir = os.path.join(output_dir, 'visualizations', self.experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Load predictions
        pred_data = np.load(self.requires()['predictions'].output().path, allow_pickle=True)
        predicted_expression = pred_data['predicted_expression']
        test_indices = pred_data['test_indices']
        gene_names = pred_data.get('gene_names', None)
        
        # Load ground truth
        data = np.load(self.requires()['data'].output().path, allow_pickle=True)
        gene_expression = data['gene_expression']
        
        # Extract ground truth for test set
        true_expression = gene_expression[test_indices]
        
        # Initialize visualization module
        vis_module = VisualizationModule(
            output_dir=os.path.join(output_dir, self.experiment_name),
            experiment_name=self.experiment_name,
            use_wandb=True
        )
        
        # Create prediction accuracy visualization
        vis_module.create_prediction_accuracy_visualization(
            predictions=predicted_expression,
            targets=true_expression,
            gene_names=gene_names
        )
        
        # Create dimensionality reduction visualizations
        vis_module.create_dimensionality_reduction_visualizations(
            predictions=predicted_expression,
            targets=true_expression,
            vis_dir=vis_dir
        )
        
        print(f"Model prediction visualizations complete. Saved to {vis_dir}")


class VisualizeDifferentialExpression(luigi.Task):
    """Luigi task for creating differential expression visualizations including volcano plots."""
    config_path = luigi.Parameter(description="Path to the configuration file")
    experiment_name = luigi.Parameter(default='default_experiment', description="Name of the experiment")
    
    def requires(self):
        from pipeline.pipeline_data_preparation import PrepareTrainingData
        return PrepareTrainingData(config_path=self.config_path)
    
    def output(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        vis_dir = os.path.join(output_dir, 'visualizations', self.experiment_name)
        return luigi.LocalTarget(os.path.join(vis_dir, 'volcano_plot.png'))
    
    def run(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        vis_dir = os.path.join(output_dir, 'visualizations', self.experiment_name)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Load training data
        data = np.load(self.requires().output().path, allow_pickle=True)
        
        # Extract gene expression and region labels
        gene_expression = data['gene_expression']
        region_labels = data.get('region_labels', None)
        gene_names = data.get('gene_names', None)
        
        # If region labels are not available, create dummy labels
        if region_labels is None:
            print("Region labels not found, creating dummy labels for demonstration...")
            n_cells = gene_expression.shape[0]
            # Assign half of cells to each region
            region_labels = np.zeros(n_cells, dtype=int)
            region_labels[n_cells//2:] = 1
        
        # Initialize visualization module
        vis_module = VisualizationModule(
            output_dir=os.path.join(output_dir, self.experiment_name),
            experiment_name=self.experiment_name,
            use_wandb=True
        )
        
        # Create volcano plot
        vis_module.create_volcano_plot(
            expression_data=gene_expression,
            region_labels=region_labels,
            gene_names=gene_names
        )
        
        print(f"Differential expression visualizations complete. Saved to {vis_dir}")


class RunVisualizationPipeline(luigi.Task):
    """Run the full visualization pipeline."""
    config_path = luigi.Parameter(description="Path to the configuration file")
    experiment_name = luigi.Parameter(default='default_experiment', description="Name of the experiment")
    
    def requires(self):
        return [
            VisualizeDataset(config_path=self.config_path),
            VisualizeModelPredictions(config_path=self.config_path, experiment_name=self.experiment_name),
            VisualizeDifferentialExpression(config_path=self.config_path, experiment_name=self.experiment_name)
        ]
    
    def output(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        output_dir = config.get('output_dir', 'output')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)
        
        return luigi.LocalTarget(os.path.join(output_dir, f"visualization_pipeline_complete_{self.experiment_name}.txt"))
    
    def run(self):
        with open(self.output().path, 'w') as f:
            f.write(f"Visualization pipeline completed successfully at {os.path.basename(self.config_path)} with experiment name {self.experiment_name}")
        
        print(f"Full visualization pipeline completed successfully for experiment {self.experiment_name}")

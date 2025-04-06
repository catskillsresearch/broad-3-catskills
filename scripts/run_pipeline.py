#!/usr/bin/env python3
# scripts/run_pipeline.py

import argparse
import luigi
import os
import yaml
import sys

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def parse_args():
    parser = argparse.ArgumentParser(description='Run the integrated spatial transcriptomics pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--experiment_name', type=str, default='default_experiment',
                        help='Name of the experiment')
    parser.add_argument('--small_dataset', action='store_true',
                        help='Use small dataset configuration')
    parser.add_argument('--step', type=str, choices=['data', 'features', 'train', 'predict', 'evaluate', 'all'],
                        default='all', help='Pipeline step to run')
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def run_pipeline(config_path, experiment_name, step='all'):
    """
    Run the integrated spatial transcriptomics pipeline.
    
    Parameters:
    - config_path: Path to configuration file
    - experiment_name: Name of the experiment
    - step: Pipeline step to run ('data', 'features', 'train', 'predict', 'evaluate', 'all')
    """
    # Import pipeline tasks
    from pipeline.pipeline_data_preparation import EnsureDirectories, CreateSmallDataset, PrepareImagePatches, ExtractMultiLevelFeatures, PrepareTrainingData
    from pipeline.pipeline_model_training import TrainModel, PredictGeneExpression, EvaluateModel, RunFullPipeline
    
    # Determine which task to run based on the step
    if step == 'data':
        task = PrepareTrainingData(config_path=config_path)
    elif step == 'features':
        task = ExtractMultiLevelFeatures(config_path=config_path)
    elif step == 'train':
        task = TrainModel(config_path=config_path, experiment_name=experiment_name)
    elif step == 'predict':
        task = PredictGeneExpression(config_path=config_path, experiment_name=experiment_name)
    elif step == 'evaluate':
        task = EvaluateModel(config_path=config_path, experiment_name=experiment_name)
    else:  # 'all'
        task = RunFullPipeline(config_path=config_path, experiment_name=experiment_name)
    
    # Run the pipeline
    luigi.build([task], local_scheduler=True)
    
    print(f"Pipeline step '{step}' completed for experiment: {experiment_name}")

if __name__ == "__main__":
    args = parse_args()
    
    # Use small dataset configuration if specified
    if args.small_dataset:
        config_path = 'config/small_dataset_config.yaml'
    else:
        config_path = args.config
    
    run_pipeline(config_path, args.experiment_name, args.step)

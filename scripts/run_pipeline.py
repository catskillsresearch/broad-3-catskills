#!/usr/bin/env python3
# scripts/run_pipeline.py

import argparse
import os
import sys
import luigi

# Add parent directory to path to allow importing modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import pipeline tasks
from pipeline.pipeline_data_preparation import PrepareTrainingData, EnsureDirectories
from pipeline.pipeline_model_training import TrainModel, PredictGeneExpression, EvaluateModel, RunFullPipeline
from pipeline.pipeline_visualization import VisualizeDataset, VisualizeModelPredictions, VisualizeDifferentialExpression, RunVisualizationPipeline

def main():
    parser = argparse.ArgumentParser(description="Run the spatial transcriptomics pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    parser.add_argument("--experiment", type=str, default="default_experiment", help="Name of the experiment")
    parser.add_argument("--step", type=str, default="all", 
                        choices=["all", "data", "model", "predict", "evaluate", "visualize"],
                        help="Pipeline step to run")
    parser.add_argument("--small_dataset", action="store_true", help="Use the small dataset for testing")
    
    args = parser.parse_args()
    
    # Determine which task to run based on the step argument
    if args.step == "data":
        task = PrepareTrainingData(config_path=args.config)
    elif args.step == "model":
        task = TrainModel(config_path=args.config, experiment_name=args.experiment)
    elif args.step == "predict":
        task = PredictGeneExpression(config_path=args.config, experiment_name=args.experiment)
    elif args.step == "evaluate":
        task = EvaluateModel(config_path=args.config, experiment_name=args.experiment)
    elif args.step == "visualize":
        task = RunVisualizationPipeline(config_path=args.config, experiment_name=args.experiment)
    else:  # "all"
        task = RunFullPipeline(config_path=args.config, experiment_name=args.experiment)
    
    # If small_dataset flag is set, modify the config path to use the small dataset config
    if args.small_dataset:
        print("Using small dataset configuration")
        # Get the directory of the config file
        config_dir = os.path.dirname(args.config)
        # Get the base name of the config file
        config_base = os.path.basename(args.config)
        # Create the small dataset config path
        small_dataset_config = os.path.join(config_dir, f"small_{config_base}")
        
        # Check if the small dataset config exists
        if os.path.exists(small_dataset_config):
            print(f"Using small dataset config: {small_dataset_config}")
            # Update the task with the small dataset config
            task = task.clone(config_path=small_dataset_config)
        else:
            print(f"Warning: Small dataset config {small_dataset_config} not found.")
            print(f"Creating a small dataset configuration from {args.config}")
            
            # Read the original config
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            
            # Modify the config for small dataset
            config['output_dir'] = os.path.join(config.get('output_dir', 'output'), 'small_dataset')
            config['small_dataset'] = True
            config['batch_size'] = min(config.get('batch_size', 32), 16)  # Reduce batch size for small dataset
            
            # Write the small dataset config
            os.makedirs(config_dir, exist_ok=True)
            with open(small_dataset_config, 'w') as f:
                yaml.dump(config, f)
            
            print(f"Created small dataset config: {small_dataset_config}")
            # Update the task with the small dataset config
            task = task.clone(config_path=small_dataset_config)
    
    # Run the Luigi task
    luigi.build([task], local_scheduler=True)

if __name__ == "__main__":
    main()

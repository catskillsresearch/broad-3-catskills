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
    
    # Run the Luigi task
    luigi.build([task], local_scheduler=True)

if __name__ == "__main__":
    main()

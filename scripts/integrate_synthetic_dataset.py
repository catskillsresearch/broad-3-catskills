#!/usr/bin/env python3
"""
Integration script for synthetic dataset functionality.
This script adds synthetic dataset support to the pipeline by creating
necessary files and updating the pipeline to work with synthetic data.
"""

import os
import sys
import shutil
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Integrate synthetic dataset functionality')
    parser.add_argument('--project_dir', type=str, default='.', help='Project root directory')
    return parser.parse_args()

def ensure_dir(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def copy_scripts(project_dir):
    """Copy synthetic dataset scripts to the project."""
    scripts_dir = os.path.join(project_dir, 'scripts')
    ensure_dir(scripts_dir)
    
    # Source files
    source_files = [
        '/home/ubuntu/scripts/create_synthetic_dataset.py',
        '/home/ubuntu/scripts/validate_synthetic_dataset.py',
        '/home/ubuntu/scripts/visualize_synthetic_dataset.py'
    ]
    
    # Copy files
    for source_file in source_files:
        if os.path.exists(source_file):
            dest_file = os.path.join(scripts_dir, os.path.basename(source_file))
            if os.path.abspath(source_file) != os.path.abspath(dest_file):
                shutil.copy(source_file, dest_file)
                print(f"Copied {source_file} to {dest_file}")
            else:
                print(f"Skipping copy of {source_file} to itself")
        else:
            print(f"Warning: Source file {source_file} not found")

def copy_config(project_dir):
    """Copy synthetic dataset configuration to the project."""
    config_dir = os.path.join(project_dir, 'config')
    ensure_dir(config_dir)
    
    # Source file
    source_file = '/home/ubuntu/broad-3-catskills/config/synthetic_config.yaml'
    
    # Copy file
    if os.path.exists(source_file):
        dest_file = os.path.join(config_dir, os.path.basename(source_file))
        if os.path.abspath(source_file) != os.path.abspath(dest_file):
            shutil.copy(source_file, dest_file)
            print(f"Copied {source_file} to {dest_file}")
        else:
            print(f"Skipping copy of {source_file} to itself")
    else:
        print(f"Warning: Source file {source_file} not found")

def copy_docs(project_dir):
    """Copy synthetic dataset documentation to the project."""
    docs_dir = os.path.join(project_dir, 'docs')
    ensure_dir(docs_dir)
    
    # Source file
    source_file = '/home/ubuntu/docs/synthetic_datasets.md'
    
    # Copy file
    if os.path.exists(source_file):
        dest_file = os.path.join(docs_dir, os.path.basename(source_file))
        if os.path.abspath(source_file) != os.path.abspath(dest_file):
            shutil.copy(source_file, dest_file)
            print(f"Copied {source_file} to {dest_file}")
        else:
            print(f"Skipping copy of {source_file} to itself")
    else:
        print(f"Warning: Source file {source_file} not found")

def update_html_docs(project_dir):
    """Update HTML documentation to include synthetic dataset information."""
    html_dir = os.path.join(project_dir, 'html')
    if not os.path.exists(html_dir):
        print(f"Warning: HTML directory {html_dir} not found, skipping HTML documentation update")
        return
    
    # Add synthetic dataset section to documentation
    pages_dir = os.path.join(html_dir, 'pages')
    if os.path.exists(pages_dir):
        # Check if synthetic_datasets.html already exists
        synthetic_html_path = os.path.join(pages_dir, 'synthetic_datasets.html')
        if not os.path.exists(synthetic_html_path):
            # Create synthetic_datasets.html
            with open(synthetic_html_path, 'w') as f:
                f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Synthetic Datasets - Spatial Transcriptomics</title>
    <link rel="stylesheet" href="../css/styles.css">
</head>
<body>
    <header>
        <h1>Synthetic Datasets for Model Validation</h1>
    </header>
    <nav>
        <ul>
            <li><a href="../index.html">Home</a></li>
            <li><a href="getting_started.html">Getting Started</a></li>
            <li><a href="data_structure.html">Data Structure</a></li>
            <li><a href="synthetic_datasets.html" class="active">Synthetic Datasets</a></li>
            <li><a href="model_architecture.html">Model Architecture</a></li>
            <li><a href="evaluation.html">Evaluation</a></li>
        </ul>
    </nav>
    <main>
        <section>
            <h2>Overview</h2>
            <p>
                Synthetic datasets with well-defined statistical properties provide a powerful approach for validating model capacity and performance. 
                Unlike real-world data, synthetic data offers complete knowledge of the underlying distributions, correlations, and patterns, 
                enabling rigorous evaluation of model capabilities.
            </p>
            <p>
                This page describes the synthetic dataset generation framework implemented in this project, including how to create synthetic datasets, 
                validate their properties, visualize the data, and use them for model training and evaluation.
            </p>
        </section>

        <section>
            <h2>Why Use Synthetic Datasets?</h2>
            <p>Synthetic datasets offer several advantages for model development and validation:</p>
            <ul>
                <li><strong>Known Ground Truth</strong>: Complete knowledge of the underlying distributions and patterns</li>
                <li><strong>Controllable Complexity</strong>: Ability to adjust difficulty levels to test model limitations</li>
                <li><strong>Reproducibility</strong>: Consistent datasets for benchmarking different models</li>
                <li><strong>Isolation of Features</strong>: Test specific aspects of model performance independently</li>
                <li><strong>No Privacy Concerns</strong>: Freedom from data privacy and sharing restrictions</li>
                <li><strong>Unlimited Data Generation</strong>: Create as much data as needed for robust testing</li>
            </ul>
        </section>

        <section>
            <h2>Creating Synthetic Datasets</h2>
            <p>
                The <code>create_synthetic_dataset.py</code> script generates synthetic spatial transcriptomics data with controllable statistical properties.
            </p>
            <h3>Basic Usage</h3>
            <pre><code>python scripts/create_synthetic_dataset.py --config config/synthetic_config.yaml --output_dir output/data/synthetic_dataset --seed 42 --visualize</code></pre>
            
            <h3>Output Structure</h3>
            <p>The synthetic dataset generator creates the following files:</p>
            <pre><code>output/data/synthetic_dataset/
├── cell_coordinates.npy      # Cell spatial coordinates
├── config.yaml               # Configuration used for generation
├── gene_expression.npy       # Gene expression matrix
├── gene_names.npy            # Gene names
├── ground_truth.npy          # Ground truth information for validation
├── metadata.yaml             # Dataset metadata
├── original_indices.npy      # Original cell indices
├── quality_scores.npy        # Cell quality scores
├── region_labels.npy         # Region assignments for cells
├── test_indices.npy          # Indices for test set
├── train_indices.npy         # Indices for training set
├── val_indices.npy           # Indices for validation set
└── visualizations/           # Visualizations of the dataset (if --visualize is used)</code></pre>
        </section>

        <section>
            <h2>Statistical Properties of Synthetic Datasets</h2>
            <p>The synthetic datasets are generated with the following controllable statistical properties:</p>
            
            <h3>1. Gene Expression Distributions</h3>
            <p>
                Gene expression values follow a negative binomial distribution, mimicking the count-based nature of real RNA-seq data. 
                The mean and dispersion parameters can be controlled to adjust the expression patterns.
            </p>
            
            <h3>2. Spatial Patterns</h3>
            <p>
                Cells are organized into spatial regions with defined boundaries. Three types of spatial patterns are supported:
            </p>
            <ul>
                <li><strong>Circular</strong>: Cells are distributed in circular regions with random centers and radii</li>
                <li><strong>Voronoi</strong>: Cells are distributed in Voronoi regions defined by random centers</li>
                <li><strong>Gradient</strong>: Cells are distributed across a continuous gradient</li>
            </ul>
            
            <h3>3. Gene Modules</h3>
            <p>
                Genes are organized into modules with correlated expression patterns. Each module has a controllable correlation strength, 
                determining how tightly the genes within the module are co-expressed.
            </p>
            
            <h3>4. Region-Specific Expression</h3>
            <p>
                Each region has a unique gene expression profile, with certain genes being differentially expressed between regions. 
                This mimics tissue-specific expression patterns in real biological samples.
            </p>
            
            <h3>5. Quality Metrics</h3>
            <p>
                Cells are assigned quality scores based on their total gene expression and spatial location. 
                Cells near the center of their region typically have higher quality scores, mimicking technical artifacts in real data.
            </p>
        </section>

        <section>
            <h2>Validating Synthetic Datasets</h2>
            <p>
                The <code>validate_synthetic_dataset.py</code> script performs a series of tests to verify that the synthetic dataset 
                has the expected statistical properties.
            </p>
            
            <h3>Basic Usage</h3>
            <pre><code>python scripts/validate_synthetic_dataset.py --dataset_dir output/data/synthetic_dataset --visualize</code></pre>
            
            <h3>Validation Tests</h3>
            <p>The validation script performs the following tests:</p>
            <ol>
                <li><strong>Distribution Tests</strong>: Verify that gene expression follows the expected negative binomial distribution</li>
                <li><strong>Spatial Tests</strong>: Check for spatial clustering of cells and spatial autocorrelation of gene expression</li>
                <li><strong>Module Tests</strong>: Validate that gene modules have the expected correlation structure</li>
                <li><strong>Region Tests</strong>: Confirm that regions have distinct gene expression profiles</li>
            </ol>
        </section>

        <section>
            <h2>Visualizing Synthetic Datasets</h2>
            <p>
                The <code>visualize_synthetic_dataset.py</code> script generates comprehensive visualizations for analyzing synthetic datasets 
                and model performance.
            </p>
            
            <h3>Basic Usage</h3>
            <pre><code>python scripts/visualize_synthetic_dataset.py --dataset_dir output/data/synthetic_dataset --predictions_dir output/predictions/synthetic_model --output_dir output/visualizations/synthetic_analysis</code></pre>
            
            <h3>Visualization Types</h3>
            <p>The visualization script generates the following types of visualizations:</p>
            <ol>
                <li><strong>Dataset Overview</strong>: General properties of the dataset, including spatial distribution, PCA, and expression distributions</li>
                <li><strong>Gene Modules</strong>: Detailed analysis of gene modules, including correlation matrices and spatial patterns</li>
                <li><strong>Region Properties</strong>: Analysis of region-specific expression patterns and differentially expressed genes</li>
                <li><strong>Model Performance</strong>: Evaluation of model predictions against ground truth, including correlation distributions and spatial patterns</li>
                <li><strong>Module Prediction</strong>: Analysis of how well the model captures gene module structure</li>
            </ol>
        </section>

        <section>
            <h2>Training Models with Synthetic Data</h2>
            <p>
                Synthetic datasets can be used for model training and evaluation just like real datasets. 
                The pipeline supports using synthetic datasets by specifying the appropriate configuration file.
            </p>
            
            <h3>Basic Usage</h3>
            <pre><code>python scripts/run_pipeline.py --config config/synthetic_config.yaml --step all</code></pre>
            
            <h3>Training Strategies</h3>
            <p>When training with synthetic data, consider the following strategies:</p>
            <ul>
                <li><strong>Complexity Progression</strong>: Start with simple synthetic datasets and gradually increase complexity</li>
                <li><strong>Feature Isolation</strong>: Create datasets that isolate specific features to test model components</li>
                <li><strong>Ablation Studies</strong>: Remove certain properties to measure their impact on model performance</li>
                <li><strong>Transfer Learning</strong>: Pre-train on synthetic data before fine-tuning on real data</li>
                <li><strong>Ensemble Approaches</strong>: Train separate models on different synthetic datasets and ensemble them</li>
            </ul>
        </section>

        <section>
            <h2>Best Practices for Using Synthetic Data</h2>
            <ol>
                <li><strong>Validate Synthetic Properties</strong>: Always validate that your synthetic dataset has the expected properties</li>
                <li><strong>Compare with Real Data</strong>: Ensure synthetic data captures key aspects of real data</li>
                <li><strong>Incremental Complexity</strong>: Start with simple synthetic datasets and gradually increase complexity</li>
                <li><strong>Diverse Synthetic Sets</strong>: Use multiple synthetic datasets with different properties</li>
                <li><strong>Document Generation Parameters</strong>: Keep detailed records of parameters used to generate each dataset</li>
                <li><strong>Benchmark Multiple Models</strong>: Use synthetic data to benchmark different model architectures</li>
                <li><strong>Combine with Real Data</strong>: Use both synthetic and real data in your workflow</li>
            </ol>
        </section>
    </main>
    <footer>
        <p>&copy; 2025 Spatial Transcriptomics Project</p>
    </footer>
    <script src="../js/script.js"></script>
</body>
</html>""")
            print(f"Created {synthetic_html_path}")
            
            # Update index.html to include link to synthetic datasets page
            index_html_path = os.path.join(html_dir, 'index.html')
            if os.path.exists(index_html_path):
                with open(index_html_path, 'r') as f:
                    content = f.read()
                
                # Check if synthetic datasets link already exists
                if '<a href="pages/synthetic_datasets.html">Synthetic Datasets</a>' not in content:
                    # Add link to navigation
                    content = content.replace(
                        '<a href="pages/data_structure.html">Data Structure</a>',
                        '<a href="pages/data_structure.html">Data Structure</a></li>\n            <li><a href="pages/synthetic_datasets.html">Synthetic Datasets</a>'
                    )
                    
                    with open(index_html_path, 'w') as f:
                        f.write(content)
                    print(f"Updated {index_html_path} with link to synthetic datasets page")
        else:
            print(f"Warning: Pages directory {pages_dir} not found, skipping HTML documentation update")

def create_run_synthetic_script(project_dir):
    """Create a script to run the synthetic dataset pipeline."""
    scripts_dir = os.path.join(project_dir, 'scripts')
    ensure_dir(scripts_dir)
    
    # Create run_synthetic_pipeline.py
    run_script_path = os.path.join(scripts_dir, 'run_synthetic_pipeline.py')
    
    # Check if file already exists
    if os.path.exists(run_script_path):
        print(f"File {run_script_path} already exists, skipping creation")
        return
        
    with open(run_script_path, 'w') as f:
        f.write("""#!/usr/bin/env python3
\"\"\"
Run the complete synthetic dataset pipeline.
This script creates a synthetic dataset, validates its properties,
trains a model on the synthetic data, and evaluates the model performance.
\"\"\"

import os
import argparse
import subprocess
import yaml

def parse_args():
    \"\"\"Parse command line arguments.\"\"\"
    parser = argparse.ArgumentParser(description='Run synthetic dataset pipeline')
    parser.add_argument('--config', type=str, default='config/synthetic_config.yaml', help='Path to configuration file')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for synthetic dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    parser.add_argument('--skip_validation', action='store_true', help='Skip dataset validation')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training')
    return parser.parse_args()

def ensure_dir(directory):
    \"\"\"Create directory if it doesn't exist.\"\"\"
    if not os.path.exists(directory):
        os.makedirs(directory)

def run_command(command):
    \"\"\"Run a command and print output.\"\"\"
    print(f"Running: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in iter(process.stdout.readline, b''):
        print(line.decode('utf-8').rstrip())
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        print(f"Command failed with return code {return_code}")
        return False
    return True

def main():
    \"\"\"Main function.\"\"\"
    args = parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Determine output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = config.get('data_path', 'output/data/synthetic_dataset')
    
    # Create synthetic dataset
    print("\\n=== Creating Synthetic Dataset ===")
    create_cmd = f"python scripts/create_synthetic_dataset.py --config {args.config} --output_dir {output_dir} --seed {args.seed}"
    if args.visualize:
        create_cmd += " --visualize"
    
    if not run_command(create_cmd):
        print("Failed to create synthetic dataset. Exiting.")
        return
    
    # Validate synthetic dataset
    if not args.skip_validation:
        print("\\n=== Validating Synthetic Dataset ===")
        validate_cmd = f"python scripts/validate_synthetic_dataset.py --dataset_dir {output_dir}"
        if args.visualize:
            validate_cmd += " --visualize"
        
        if not run_command(validate_cmd):
            print("Warning: Dataset validation failed. Continuing with pipeline.")
    
    # Run pipeline with synthetic dataset
    if not args.skip_training:
        print("\\n=== Running Pipeline with Synthetic Dataset ===")
        pipeline_cmd = f"python scripts/run_pipeline.py --config {args.config} --step all"
        
        if not run_command(pipeline_cmd):
            print("Warning: Pipeline execution failed.")
    
    # Generate visualizations of model performance
    if args.visualize and not args.skip_training:
        print("\\n=== Generating Model Performance Visualizations ===")
        experiment_name = config.get('experiment_name', 'default_experiment')
        predictions_dir = os.path.join(config.get('output_dir', 'output'), 'predictions', experiment_name)
        vis_output_dir = os.path.join(config.get('output_dir', 'output'), 'visualizations', 'synthetic_analysis')
        
        visualize_cmd = f"python scripts/visualize_synthetic_dataset.py --dataset_dir {output_dir} --predictions_dir {predictions_dir} --output_dir {vis_output_dir} --analysis_type all"
        
        if not run_command(visualize_cmd):
            print("Warning: Visualization generation failed.")
    
    print("\\n=== Synthetic Dataset Pipeline Complete ===")

if __name__ == "__main__":
    main()
""")
    print(f"Created {run_script_path}")
    
    # Make script executable
    os.chmod(run_script_path, 0o755)

def main():
    """Main function."""
    args = parse_args()
    project_dir = args.project_dir
    
    print(f"Integrating synthetic dataset functionality into {project_dir}...")
    
    # Copy scripts
    copy_scripts(project_dir)
    
    # Copy config
    copy_config(project_dir)
    
    # Copy docs
    copy_docs(project_dir)
    
    # Update HTML docs
    update_html_docs(project_dir)
    
    # Create run synthetic script
    create_run_synthetic_script(project_dir)
    
    print("Integration complete!")
    print("\nTo use synthetic datasets:")
    print("1. Create a synthetic dataset:")
    print("   python scripts/create_synthetic_dataset.py --config config/synthetic_config.yaml --output_dir output/data/synthetic_dataset --visualize")
    print("2. Validate the synthetic dataset:")
    print("   python scripts/validate_synthetic_dataset.py --dataset_dir output/data/synthetic_dataset --visualize")
    print("3. Run the complete synthetic pipeline:")
    print("   python scripts/run_synthetic_pipeline.py --config config/synthetic_config.yaml --visualize")
    print("\nSee docs/synthetic_datasets.md for detailed documentation.")

if __name__ == "__main__":
    main()

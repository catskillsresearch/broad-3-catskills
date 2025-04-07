# Synthetic Datasets for Model Validation

## Overview

Synthetic datasets with well-defined statistical properties provide a powerful approach for validating model capacity and performance. Unlike real-world data, synthetic data offers complete knowledge of the underlying distributions, correlations, and patterns, enabling rigorous evaluation of model capabilities.

This documentation describes the synthetic dataset generation framework implemented in this project, including how to create synthetic datasets, validate their properties, visualize the data, and use them for model training and evaluation.

## Why Use Synthetic Datasets?

Synthetic datasets offer several advantages for model development and validation:

1. **Known Ground Truth**: Complete knowledge of the underlying distributions and patterns
2. **Controllable Complexity**: Ability to adjust difficulty levels to test model limitations
3. **Reproducibility**: Consistent datasets for benchmarking different models
4. **Isolation of Features**: Test specific aspects of model performance independently
5. **No Privacy Concerns**: Freedom from data privacy and sharing restrictions
6. **Unlimited Data Generation**: Create as much data as needed for robust testing

## Creating Synthetic Datasets

The `create_synthetic_dataset.py` script generates synthetic spatial transcriptomics data with controllable statistical properties.

### Basic Usage

```bash
python scripts/create_synthetic_dataset.py --config config/synthetic_config.yaml --output_dir output/data/synthetic_dataset --seed 42 --visualize
```

### Configuration Parameters

The synthetic dataset generator accepts the following configuration parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| n_cells | Number of cells to generate | 2000 |
| n_genes | Number of genes to generate | 200 |
| n_regions | Number of spatial regions | 5 |
| region_type | Type of regions ('circular', 'voronoi', 'gradient') | 'circular' |
| space_size | Size of spatial domain | 1000 |
| n_gene_modules | Number of gene modules | 10 |
| min_module_size | Minimum genes per module | 5 |
| max_module_size | Maximum genes per module | 20 |
| base_expression | Base expression level | 5.0 |
| expression_scale | Scale of expression variation | 2.0 |
| noise_level | Level of random noise | 0.2 |
| spatial_effect_strength | Strength of spatial effects | 0.5 |
| nb_dispersion | Negative binomial dispersion parameter | 0.1 |
| quality_noise | Noise in quality scores | 0.1 |
| val_test_size | Fraction of data for validation and testing | 0.3 |

### Example Configuration

```yaml
# config/synthetic_config.yaml
output_dir: output
data_path: output/data/synthetic_dataset

# Synthetic dataset parameters
n_cells: 2000
n_genes: 200
n_regions: 5
region_type: circular
space_size: 1000
n_gene_modules: 10
min_module_size: 5
max_module_size: 20
base_expression: 5.0
expression_scale: 2.0
noise_level: 0.2
spatial_effect_strength: 0.5
nb_dispersion: 0.1
quality_noise: 0.1
val_test_size: 0.3

# Model parameters
model_type: mlp
hidden_layers: [512, 256]
dropout: 0.2
learning_rate: 0.001
batch_size: 32
epochs: 100
```

### Output Structure

The synthetic dataset generator creates the following files:

```
output/data/synthetic_dataset/
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
└── visualizations/           # Visualizations of the dataset (if --visualize is used)
```

## Statistical Properties of Synthetic Datasets

The synthetic datasets are generated with the following controllable statistical properties:

### 1. Gene Expression Distributions

Gene expression values follow a negative binomial distribution, mimicking the count-based nature of real RNA-seq data. The mean and dispersion parameters can be controlled to adjust the expression patterns.

### 2. Spatial Patterns

Cells are organized into spatial regions with defined boundaries. Three types of spatial patterns are supported:

- **Circular**: Cells are distributed in circular regions with random centers and radii
- **Voronoi**: Cells are distributed in Voronoi regions defined by random centers
- **Gradient**: Cells are distributed across a continuous gradient

### 3. Gene Modules

Genes are organized into modules with correlated expression patterns. Each module has a controllable correlation strength, determining how tightly the genes within the module are co-expressed.

### 4. Region-Specific Expression

Each region has a unique gene expression profile, with certain genes being differentially expressed between regions. This mimics tissue-specific expression patterns in real biological samples.

### 5. Quality Metrics

Cells are assigned quality scores based on their total gene expression and spatial location. Cells near the center of their region typically have higher quality scores, mimicking technical artifacts in real data.

## Validating Synthetic Datasets

The `validate_synthetic_dataset.py` script performs a series of tests to verify that the synthetic dataset has the expected statistical properties.

### Basic Usage

```bash
python scripts/validate_synthetic_dataset.py --dataset_dir output/data/synthetic_dataset --visualize
```

### Validation Tests

The validation script performs the following tests:

1. **Distribution Tests**: Verify that gene expression follows the expected negative binomial distribution
2. **Spatial Tests**: Check for spatial clustering of cells and spatial autocorrelation of gene expression
3. **Module Tests**: Validate that gene modules have the expected correlation structure
4. **Region Tests**: Confirm that regions have distinct gene expression profiles

### Validation Output

The validation results are saved to:

```
output/data/synthetic_dataset/validation/
├── validation_results.yaml   # Quantitative validation results
└── [visualization files]     # Visualizations of validation tests (if --visualize is used)
```

## Visualizing Synthetic Datasets

The `visualize_synthetic_dataset.py` script generates comprehensive visualizations for analyzing synthetic datasets and model performance.

### Basic Usage

```bash
python scripts/visualize_synthetic_dataset.py --dataset_dir output/data/synthetic_dataset --predictions_dir output/predictions/synthetic_model --output_dir output/visualizations/synthetic_analysis
```

### Visualization Types

The visualization script generates the following types of visualizations:

1. **Dataset Overview**: General properties of the dataset, including spatial distribution, PCA, and expression distributions
2. **Gene Modules**: Detailed analysis of gene modules, including correlation matrices and spatial patterns
3. **Region Properties**: Analysis of region-specific expression patterns and differentially expressed genes
4. **Model Performance**: Evaluation of model predictions against ground truth, including correlation distributions and spatial patterns
5. **Module Prediction**: Analysis of how well the model captures gene module structure

## Training Models with Synthetic Data

Synthetic datasets can be used for model training and evaluation just like real datasets. The pipeline supports using synthetic datasets by specifying the appropriate configuration file.

### Basic Usage

```bash
python scripts/run_pipeline.py --config config/synthetic_config.yaml --step all
```

### Training Strategies

When training with synthetic data, consider the following strategies:

1. **Complexity Progression**: Start with simple synthetic datasets and gradually increase complexity
2. **Feature Isolation**: Create datasets that isolate specific features to test model components
3. **Ablation Studies**: Remove certain properties to measure their impact on model performance
4. **Transfer Learning**: Pre-train on synthetic data before fine-tuning on real data
5. **Ensemble Approaches**: Train separate models on different synthetic datasets and ensemble them

## Evaluating Model Performance with Synthetic Data

Synthetic datasets provide unique opportunities for model evaluation due to the known ground truth.

### Evaluation Metrics

When evaluating models on synthetic data, consider these metrics:

1. **Expression Accuracy**: How well the model predicts gene expression values
2. **Correlation Structure**: How well the model captures gene-gene correlations
3. **Spatial Pattern Recovery**: How well the model recovers spatial expression patterns
4. **Module Detection**: How well the model identifies gene modules
5. **Region Discrimination**: How well the model distinguishes between regions

### Visualization of Model Performance

The visualization script provides detailed visualizations of model performance on synthetic data:

```bash
python scripts/visualize_synthetic_dataset.py --dataset_dir output/data/synthetic_dataset --predictions_dir output/predictions/synthetic_model --analysis_type model
```

## Best Practices for Using Synthetic Data

1. **Validate Synthetic Properties**: Always validate that your synthetic dataset has the expected properties
2. **Compare with Real Data**: Ensure synthetic data captures key aspects of real data
3. **Incremental Complexity**: Start with simple synthetic datasets and gradually increase complexity
4. **Diverse Synthetic Sets**: Use multiple synthetic datasets with different properties
5. **Document Generation Parameters**: Keep detailed records of parameters used to generate each dataset
6. **Benchmark Multiple Models**: Use synthetic data to benchmark different model architectures
7. **Combine with Real Data**: Use both synthetic and real data in your workflow

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce dataset size (n_cells, n_genes) if encountering memory issues
2. **Visualization Errors**: Some visualizations require specific packages; install umap-learn for UMAP visualizations
3. **Unexpected Distributions**: Adjust distribution parameters if the synthetic data doesn't match expected patterns
4. **Pipeline Integration**: Ensure synthetic dataset paths are correctly specified in configuration files

### Debugging Tips

1. Use the `--visualize` flag to generate visualizations for debugging
2. Check the metadata.yaml file for dataset properties
3. Examine the ground_truth.npy file for expected patterns
4. Run validation tests to verify dataset properties

## Extending the Synthetic Framework

The synthetic dataset framework can be extended in several ways:

1. **New Distribution Types**: Implement additional statistical distributions for gene expression
2. **Complex Spatial Patterns**: Add more sophisticated spatial patterns (e.g., branching structures)
3. **Temporal Dynamics**: Extend to include time-series data with temporal patterns
4. **Multi-omics Integration**: Generate coordinated synthetic datasets for multiple omics types
5. **Disease Models**: Create synthetic models of disease states with known perturbations

## Conclusion

Synthetic datasets with well-defined statistical properties provide a powerful approach for validating model capacity and performance. By using synthetic data alongside real data, you can gain deeper insights into model behavior, identify limitations, and develop more robust computational methods.

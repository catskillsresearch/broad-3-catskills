# Revised Implementation Documentation

## Overview

This document provides detailed information about the revised implementation of the spatial transcriptomics analysis pipeline. The implementation now integrates three distinct approaches from the Broad Institute IBD Challenge:

1. **DeepSpot (Crunch 1)**: For predicting measured gene expression
2. **Tarandros (Crunch 2)**: For predicting unmeasured gene expression with a focus on cell-wise Spearman correlation
3. **logFC Method (Crunch 3)**: For gene ranking to identify markers of pre-cancerous tissue regions

## Implementation Structure

The revised implementation consists of the following components:

1. **DeepSpot Model** (`models/deepspot_model.py`) - From original implementation
2. **Tarandros Model** (`models/tarandros_model.py`) - New implementation
3. **LogFC Gene Ranking** (`models/logfc_gene_ranking.py`) - New implementation
4. **Integrated Pipeline** (`models/integrated_pipeline.py`) - New implementation
5. **Testing Script** (`test_integrated_pipeline.py`) - New implementation

## Key Changes from Original Implementation

### 1. Prioritizing Cell-wise Spearman Correlation

The original implementation focused primarily on gene-wise Spearman correlation using the DeepSpot architecture. The revised implementation adds the Tarandros approach which prioritizes cell-wise Spearman correlation:

- Implemented `cell_wise_spearman_loss` function that specifically optimizes for cell-wise correlation
- Created cell-type specific embedding and attention mechanisms
- Added spatial context aggregation that preserves local relationships
- Implemented similarity-based refinement using reference scRNA-seq data

### 2. Integration of logFC Method for Gene Ranking

The revised implementation adds the logFC method from Crunch 3 for gene ranking:

- Implemented `LogFCGeneRanking` class for computing log fold change between dysplastic and non-dysplastic regions
- Added functionality to rank genes based on absolute logFC values
- Included visualization capabilities for gene rankings
- Implemented feature selection based on gene rankings

### 3. Comprehensive Integrated Pipeline

The revised implementation provides a unified pipeline that combines all three approaches:

- Created `IntegratedPipeline` class that runs all three crunches sequentially
- Implemented `IntegratedLightningModule` for training with PyTorch Lightning
- Added weighted loss functions that balance gene-wise and cell-wise metrics
- Provided comprehensive evaluation and visualization capabilities

## Tarandros Model Architecture

The Tarandros model architecture is designed to prioritize cell-wise Spearman correlation and includes:

### 1. Cell-wise Attention

The `CellWiseAttention` module applies attention across genes for each cell, allowing the model to focus on the most informative genes for each cell's expression profile.

### 2. Cell Type Embedding

The `CellTypeEmbedding` module predicts cell types and generates cell-specific embeddings, helping to capture cell-specific patterns in gene expression.

### 3. Spatial Context Aggregation

The `SpatialContextAggregation` module preserves local spatial relationships by applying distance-based weighting to neighboring cells.

### 4. Similarity-based Refinement

The model uses similarity between measured gene expressions and reference data to refine predictions for unmeasured genes, leveraging prior knowledge from scRNA-seq data.

## LogFC Gene Ranking

The logFC gene ranking approach includes:

### 1. Log Fold Change Computation

The `compute_logfc` method calculates the log2 fold change between mean expression values in dysplastic and non-dysplastic regions.

### 2. Gene Ranking

The `rank_genes` method ranks genes based on the absolute magnitude of their logFC values, with higher values indicating greater differentiation potential.

### 3. Feature Selection

The `LogFCBasedFeatureSelection` module allows for selecting the most informative genes based on their rankings for downstream tasks.

## Integrated Pipeline

The integrated pipeline combines all three approaches:

### 1. Sequential Processing

The pipeline first uses DeepSpot to predict measured gene expression, then uses Tarandros to predict unmeasured gene expression, and finally applies logFC ranking to identify marker genes.

### 2. Balanced Loss Functions

The pipeline uses a weighted combination of gene-wise and cell-wise Spearman correlation losses, with configurable weights to balance the two metrics.

### 3. Comprehensive Evaluation

The pipeline provides evaluation metrics for both gene-wise and cell-wise performance, allowing for a complete assessment of model capabilities.

## Configuration Parameters

The revised implementation uses the following configuration parameters:

- **feature_dim**: Dimension of input features (512)
- **embedding_dim**: Dimension of integrated features (256)
- **n_cell_types**: Number of cell types for Tarandros model (10)
- **n_measured_genes**: Number of measured genes (460)
- **n_unmeasured_genes**: Number of unmeasured genes (18157)
- **gene_wise_weight**: Weight for gene-wise Spearman loss (0.3)
- **cell_wise_weight**: Weight for cell-wise Spearman loss (0.7)
- **learning_rate**: Learning rate for optimizer (0.001)
- **weight_decay**: Weight decay for optimizer (0.0001)

## Testing

The implementation includes a comprehensive testing script (`test_integrated_pipeline.py`) that validates:

1. The DeepSpot model for Crunch 1
2. The Tarandros model for Crunch 2
3. The LogFC ranking for Crunch 3
4. The integrated pipeline combining all three approaches

The tests verify that each component produces outputs with the expected shapes and that the integrated pipeline correctly combines all three approaches.

## Usage

To use the integrated pipeline:

```python
from broad_3_catskills.models.integrated_pipeline import IntegratedPipeline

# Create pipeline
pipeline = IntegratedPipeline(config)

# Run full pipeline
results = pipeline.run_full_pipeline(
    spot_features,
    subspot_features,
    neighbor_features,
    neighbor_distances,
    measured_gene_expression,
    cell_labels,
    measured_gene_names,
    unmeasured_gene_names,
    reference_data
)

# Access results
crunch1_predictions = results['crunch1_predictions']
crunch2_predictions = results['crunch2_predictions']
combined_expression = results['combined_expression']
gene_rankings = results['gene_rankings']
```

## Conclusion

This revised implementation provides a comprehensive solution for spatial transcriptomics analysis that combines the strengths of all three approaches from the Broad Institute IBD Challenge:

- DeepSpot's multi-level spatial context integration for predicting measured genes
- Tarandros's cell-focused approach for better cell-wise Spearman correlation
- logFC method for identifying marker genes of pre-cancerous tissue regions

The implementation prioritizes cell-wise Spearman correlation in the Crunch 2 part as requested, while maintaining the strengths of the original DeepSpot architecture and incorporating the logFC method for gene ranking.

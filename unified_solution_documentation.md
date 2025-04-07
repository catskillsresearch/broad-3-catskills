# Unified Solution Documentation

## Overview

This document provides detailed information about the unified solution for spatial transcriptomics analysis. This implementation combines the sophisticated neural network architecture from the original approach with the comprehensive framework of the revised implementation, creating a single integrated solution that leverages the best aspects of both approaches.

## Key Components

The unified solution consists of the following key components:

### 1. Advanced Neural Network Architecture

- **GraphAttentionLayer**: Sophisticated graph attention network for modeling gene-gene relationships
- **SpatialAttention**: Advanced attention mechanism for integrating multi-level spatial context
- **EnhancedCellEmbedding**: Transformer-like architecture with multi-head attention for cell embedding
- **CellTypeEmbedding**: Cell type prediction and embedding for cell-specific patterns

### 2. Comprehensive Pipeline Framework

- **UnifiedDeepSpotModel**: Core model that combines DeepSpot and Tarandros approaches
- **UnifiedPipeline**: Complete pipeline that runs all three crunches sequentially
- **UnifiedLightningModule**: PyTorch Lightning module for training and evaluation
- **LogFCGeneRanking**: Implementation of the logFC method for gene ranking

### 3. Specialized Loss Functions

- **spearman_correlation_loss**: Differentiable approximation of Spearman correlation loss
- **cell_wise_spearman_loss**: Cell-wise focused Spearman correlation loss

## Architecture Details

### UnifiedDeepSpotModel

The `UnifiedDeepSpotModel` is the core of the unified solution, combining the best aspects of DeepSpot and Tarandros approaches:

1. **Multi-level Feature Processing**:
   - Processes spot-level features with `phi_spot`
   - Processes sub-spot features with `phi_subspot`
   - Processes neighbor features with `phi_neighbor`

2. **Spatial Context Integration**:
   - Uses `SpatialAttention` to integrate sub-spot context
   - Uses `SpatialAttention` with distance weighting to integrate neighbor context

3. **Cell-specific Processing**:
   - Uses `CellTypeEmbedding` to predict cell types and generate cell-specific embeddings
   - Uses `EnhancedCellEmbedding` with transformer-like architecture for sophisticated cell representation

4. **Gene-gene Relationship Modeling**:
   - Uses `GraphAttentionLayer` to model relationships between genes
   - Refines gene predictions based on gene-gene relationships

5. **Two-stage Prediction**:
   - First predicts measured genes using cell embeddings
   - Then predicts unmeasured genes using both cell embeddings and measured gene predictions

6. **Reference-based Refinement**:
   - Uses similarity between predicted measured gene expressions and reference data
   - Refines unmeasured gene predictions based on cell type probabilities and expression similarity

### UnifiedPipeline

The `UnifiedPipeline` provides a comprehensive framework for running the full analysis:

1. **Crunch 1 & 2 Integration**:
   - Runs both measured and unmeasured gene prediction in a single step
   - Uses the unified model for both tasks
   - Evaluates both gene-wise and cell-wise Spearman correlation

2. **Crunch 3 Implementation**:
   - Uses `LogFCGeneRanking` to rank genes based on log fold change
   - Identifies genes that differentiate between dysplastic and non-dysplastic regions

3. **Full Pipeline Integration**:
   - Runs all three crunches sequentially
   - Combines results into a comprehensive output

### UnifiedLightningModule

The `UnifiedLightningModule` provides a PyTorch Lightning interface for training and evaluation:

1. **Balanced Loss Function**:
   - Combines gene-wise and cell-wise Spearman correlation losses
   - Uses configurable weights to balance the two metrics

2. **Comprehensive Metrics Logging**:
   - Logs separate metrics for measured and unmeasured genes
   - Logs both gene-wise and cell-wise performance

3. **Optimizer and Scheduler Configuration**:
   - Uses AdamW optimizer with configurable learning rate and weight decay
   - Uses ReduceLROnPlateau scheduler for adaptive learning rate

## Implementation Highlights

### 1. Sophisticated Attention Mechanisms

The unified solution uses advanced attention mechanisms throughout:

```python
# Spatial attention for context integration
scores = torch.matmul(q, k.transpose(-2, -1)) / (self.feature_dim ** 0.5)
if distances is not None:
    distance_weights = 1.0 / (distances.unsqueeze(1) + 1.0)
    scores = scores * distance_weights
attn_weights = F.softmax(scores, dim=-1)
context = torch.matmul(attn_weights, v).squeeze(1)
```

### 2. Graph Attention for Gene-Gene Relationships

The solution models gene-gene relationships using graph attention networks:

```python
# Apply graph attention to refine predictions
if self.use_gene_graph:
    initial_predictions = self.measured_gene_predictor(cell_embedding)
    measured_predictions = initial_predictions + 0.1 * self.gene_graph_layer(
        initial_predictions, self.gene_adj
    )
```

### 3. Cell-wise Optimization

The solution prioritizes cell-wise Spearman correlation through specialized components and loss functions:

```python
# Cell-wise Spearman loss
pred_ranks = _to_ranks(predictions)
target_ranks = _to_ranks(targets)
pred_mean = pred_ranks.mean(dim=1, keepdim=True)
target_mean = target_ranks.mean(dim=1, keepdim=True)
pred_diff = pred_ranks - pred_mean
target_diff = target_ranks - target_mean
cov = (pred_diff * target_diff).sum(dim=1)
pred_std = torch.sqrt((pred_diff ** 2).sum(dim=1) + eps)
target_std = torch.sqrt((target_diff ** 2).sum(dim=1) + eps)
correlation = cov / (pred_std * target_std + eps)
return 1.0 - correlation.mean()
```

### 4. Reference-based Refinement

The solution uses reference data to refine unmeasured gene predictions:

```python
# Apply similarity-based refinement using reference data
measured_similarity = self._compute_expression_similarity(
    measured_predictions, 
    self.reference_measured_expressions
)
cell_type_probs = F.softmax(cell_type_logits, dim=-1)
combined_weights = cell_type_probs * measured_similarity
combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)
reference_predictions = torch.matmul(combined_weights, self.reference_expressions)
unmeasured_predictions = alpha * unmeasured_predictions + (1 - alpha) * reference_predictions
```

## Configuration Parameters

The unified solution uses the following configuration parameters:

- **feature_dim**: Dimension of input features (512)
- **phi_dim**: Dimension of processed features (256)
- **embedding_dim**: Dimension of cell embeddings (512)
- **n_heads**: Number of attention heads (4)
- **n_cell_types**: Number of cell types (10)
- **n_measured_genes**: Number of measured genes (460)
- **n_unmeasured_genes**: Number of unmeasured genes (18157)
- **gene_wise_weight**: Weight for gene-wise Spearman loss (0.3)
- **cell_wise_weight**: Weight for cell-wise Spearman loss (0.7)
- **learning_rate**: Learning rate for optimizer (0.001)
- **weight_decay**: Weight decay for optimizer (0.0001)
- **dropout**: Dropout rate (0.3)

## Usage

### Basic Usage

```python
from broad_3_catskills.models.unified_approach import UnifiedPipeline

# Create pipeline
pipeline = UnifiedPipeline(config)

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
measured_predictions = results['measured_predictions']
unmeasured_predictions = results['unmeasured_predictions']
combined_expression = results['combined_expression']
gene_rankings = results['gene_rankings']
```

### Training with PyTorch Lightning

```python
from broad_3_catskills.models.unified_approach import UnifiedLightningModule
import pytorch_lightning as pl

# Create model
model = UnifiedLightningModule(config)

# Create trainer
trainer = pl.Trainer(
    max_epochs=100,
    gpus=1,
    callbacks=[
        pl.callbacks.EarlyStopping(monitor='val_loss', patience=10),
        pl.callbacks.ModelCheckpoint(monitor='val_loss')
    ]
)

# Train model
trainer.fit(model, train_dataloader, val_dataloader)

# Test model
trainer.test(model, test_dataloader)
```

## Advantages Over Previous Implementations

### Compared to Original Implementation

1. **More Comprehensive Pipeline**: Includes all three crunches in a unified framework
2. **Better Cell-wise Optimization**: Prioritizes cell-wise Spearman correlation
3. **Reference-based Refinement**: Uses reference data to improve unmeasured gene predictions
4. **Gene Ranking**: Includes logFC method for gene ranking

### Compared to Revised Implementation

1. **More Sophisticated Architecture**: Uses advanced attention mechanisms and transformer-like components
2. **Better Gene-Gene Relationship Modeling**: Uses graph attention networks
3. **More Integrated Approach**: Combines all components in a single model
4. **More Efficient Pipeline**: Runs Crunch 1 and 2 in a single step

## Conclusion

This unified solution combines the best aspects of both previous implementations:

- The sophisticated neural network architecture from the original implementation
- The comprehensive framework and cell-wise optimization from the revised implementation

The result is a single, integrated solution that provides state-of-the-art performance for spatial transcriptomics analysis, with a focus on both gene-wise and cell-wise metrics, and a complete pipeline for all three crunches of the Broad Institute IBD Challenge.

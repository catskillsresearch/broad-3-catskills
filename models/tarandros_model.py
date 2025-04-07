#!/usr/bin/env python3
# models/tarandros_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

class CellWiseAttention(nn.Module):
    """
    Cell-wise attention mechanism that prioritizes cell-wise Spearman correlation.
    This is a key component of the Tarandros approach from Crunch 2.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = np.sqrt(hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply cell-wise attention to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, n_genes, feature_dim)
            
        Returns:
            Tensor of shape (batch_size, n_genes, hidden_dim)
        """
        # Compute query, key, value
        q = self.query(x)  # (batch_size, n_genes, hidden_dim)
        k = self.key(x)    # (batch_size, n_genes, hidden_dim)
        v = self.value(x)  # (batch_size, n_genes, hidden_dim)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # (batch_size, n_genes, n_genes)
        
        # Apply softmax to get attention weights
        weights = F.softmax(scores, dim=-1)  # (batch_size, n_genes, n_genes)
        
        # Apply attention weights to values
        output = torch.matmul(weights, v)  # (batch_size, n_genes, hidden_dim)
        
        return output

class CellTypeEmbedding(nn.Module):
    """
    Cell type embedding module that helps capture cell-specific patterns.
    This is a key component of the Tarandros approach from Crunch 2.
    """
    def __init__(self, input_dim: int, embedding_dim: int, n_cell_types: int = 10):
        super().__init__()
        self.embedding = nn.Embedding(n_cell_types, embedding_dim)
        self.cell_type_predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_cell_types)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict cell types and get cell type embeddings.
        
        Args:
            x: Input tensor of shape (batch_size, feature_dim)
            
        Returns:
            Tuple of (cell_type_logits, cell_embeddings)
        """
        # Predict cell types
        cell_type_logits = self.cell_type_predictor(x)  # (batch_size, n_cell_types)
        
        # Get soft cell type assignments
        cell_type_probs = F.softmax(cell_type_logits, dim=-1)  # (batch_size, n_cell_types)
        
        # Get embeddings for each cell type
        type_embeddings = self.embedding.weight  # (n_cell_types, embedding_dim)
        
        # Compute weighted sum of embeddings based on cell type probabilities
        cell_embeddings = torch.matmul(cell_type_probs, type_embeddings)  # (batch_size, embedding_dim)
        
        return cell_type_logits, cell_embeddings

class SpatialContextAggregation(nn.Module):
    """
    Spatial context aggregation module that preserves local spatial relationships.
    This is a key component of the Tarandros approach from Crunch 2.
    """
    def __init__(self, feature_dim: int, context_dim: int):
        super().__init__()
        self.feature_transform = nn.Linear(feature_dim, context_dim)
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, context_dim)
        )
        self.context_integration = nn.Sequential(
            nn.Linear(context_dim * 2, context_dim),
            nn.ReLU(),
            nn.Linear(context_dim, context_dim)
        )
        
    def forward(self, 
                features: torch.Tensor, 
                neighbor_features: torch.Tensor, 
                neighbor_distances: torch.Tensor) -> torch.Tensor:
        """
        Aggregate spatial context from neighboring cells.
        
        Args:
            features: Cell features of shape (batch_size, feature_dim)
            neighbor_features: Neighbor features of shape (batch_size, n_neighbors, feature_dim)
            neighbor_distances: Neighbor distances of shape (batch_size, n_neighbors)
            
        Returns:
            Aggregated context features of shape (batch_size, context_dim)
        """
        batch_size, n_neighbors, _ = neighbor_features.shape
        
        # Transform features
        transformed_neighbors = self.feature_transform(neighbor_features)  # (batch_size, n_neighbors, context_dim)
        
        # Embed distances
        distances = neighbor_distances.unsqueeze(-1)  # (batch_size, n_neighbors, 1)
        distance_weights = self.distance_embedding(distances)  # (batch_size, n_neighbors, context_dim)
        
        # Apply distance-based weighting
        weighted_neighbors = transformed_neighbors * torch.sigmoid(distance_weights)
        
        # Aggregate neighbor features (weighted average)
        aggregated_context = weighted_neighbors.mean(dim=1)  # (batch_size, context_dim)
        
        # Transform cell features
        cell_context = self.feature_transform(features)  # (batch_size, context_dim)
        
        # Integrate cell and neighbor context
        combined = torch.cat([cell_context, aggregated_context], dim=-1)  # (batch_size, context_dim*2)
        integrated_context = self.context_integration(combined)  # (batch_size, context_dim)
        
        return integrated_context

class TarandrosModel(nn.Module):
    """
    Tarandros model that prioritizes cell-wise Spearman correlation.
    This model is designed for Crunch 2 to predict unmeasured genes.
    """
    def __init__(self, config: Dict):
        super().__init__()
        # Extract configuration parameters
        self.feature_dim = config.get('feature_dim', 512)
        self.embedding_dim = config.get('embedding_dim', 256)
        self.n_cell_types = config.get('n_cell_types', 10)
        self.n_measured_genes = config.get('n_measured_genes', 460)
        self.n_unmeasured_genes = config.get('n_unmeasured_genes', 18157)
        
        # Cell type embedding
        self.cell_type_module = CellTypeEmbedding(
            input_dim=self.feature_dim,
            embedding_dim=self.embedding_dim,
            n_cell_types=self.n_cell_types
        )
        
        # Spatial context aggregation
        self.spatial_context = SpatialContextAggregation(
            feature_dim=self.feature_dim,
            context_dim=self.embedding_dim
        )
        
        # Cell-wise attention for gene expression
        self.gene_attention = CellWiseAttention(
            input_dim=self.n_measured_genes,
            hidden_dim=self.embedding_dim
        )
        
        # Gene expression prediction
        self.gene_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.n_unmeasured_genes)
        )
        
        # Reference gene expression database (to be initialized with scRNA-seq data)
        self.register_buffer('reference_expressions', torch.zeros(self.n_cell_types, self.n_unmeasured_genes))
        self.register_buffer('reference_measured_expressions', torch.zeros(self.n_cell_types, self.n_measured_genes))
        
    def set_reference_data(self, 
                          measured_expressions: torch.Tensor, 
                          unmeasured_expressions: torch.Tensor):
        """
        Set reference gene expression data from scRNA-seq.
        
        Args:
            measured_expressions: Expression of measured genes for each cell type (n_cell_types, n_measured_genes)
            unmeasured_expressions: Expression of unmeasured genes for each cell type (n_cell_types, n_unmeasured_genes)
        """
        self.reference_measured_expressions = measured_expressions
        self.reference_expressions = unmeasured_expressions
        
    def forward(self, 
               features: torch.Tensor, 
               measured_expressions: torch.Tensor,
               neighbor_features: Optional[torch.Tensor] = None,
               neighbor_distances: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the Tarandros model.
        
        Args:
            features: Cell features of shape (batch_size, feature_dim)
            measured_expressions: Measured gene expressions of shape (batch_size, n_measured_genes)
            neighbor_features: Neighbor features of shape (batch_size, n_neighbors, feature_dim)
            neighbor_distances: Neighbor distances of shape (batch_size, n_neighbors)
            
        Returns:
            Predicted unmeasured gene expressions of shape (batch_size, n_unmeasured_genes)
        """
        batch_size = features.shape[0]
        
        # Get cell type embeddings
        cell_type_logits, cell_embeddings = self.cell_type_module(features)
        
        # Get spatial context if neighbors are provided
        if neighbor_features is not None and neighbor_distances is not None:
            spatial_embeddings = self.spatial_context(
                features, neighbor_features, neighbor_distances
            )
        else:
            spatial_embeddings = torch.zeros(batch_size, self.embedding_dim, device=features.device)
        
        # Apply cell-wise attention to measured gene expressions
        measured_expressions_expanded = measured_expressions.unsqueeze(-1).expand(-1, -1, self.embedding_dim)
        gene_embeddings = self.gene_attention(measured_expressions_expanded).mean(dim=1)
        
        # Combine embeddings
        combined_embeddings = torch.cat([
            cell_embeddings, 
            spatial_embeddings,
            gene_embeddings
        ], dim=-1)
        
        # Predict unmeasured gene expressions
        predicted_expressions = self.gene_predictor(combined_embeddings)
        
        # Apply similarity-based refinement using reference data
        cell_type_probs = F.softmax(cell_type_logits, dim=-1)  # (batch_size, n_cell_types)
        
        # Compute similarity between measured expressions and reference measured expressions
        measured_similarity = self._compute_expression_similarity(
            measured_expressions, 
            self.reference_measured_expressions
        )  # (batch_size, n_cell_types)
        
        # Combine cell type probabilities and expression similarity
        combined_weights = cell_type_probs * measured_similarity
        combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # Get reference-based predictions
        reference_predictions = torch.matmul(combined_weights, self.reference_expressions)
        
        # Blend model predictions with reference-based predictions
        alpha = 0.7  # Weight for model predictions vs. reference-based predictions
        final_predictions = alpha * predicted_expressions + (1 - alpha) * reference_predictions
        
        return final_predictions
    
    def _compute_expression_similarity(self, 
                                      expressions: torch.Tensor, 
                                      reference_expressions: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between expressions and reference expressions.
        
        Args:
            expressions: Gene expressions of shape (batch_size, n_genes)
            reference_expressions: Reference gene expressions of shape (n_cell_types, n_genes)
            
        Returns:
            Similarity scores of shape (batch_size, n_cell_types)
        """
        # Normalize expressions
        norm_expr = F.normalize(expressions, p=2, dim=-1)
        norm_ref = F.normalize(reference_expressions, p=2, dim=-1)
        
        # Compute cosine similarity
        similarity = torch.matmul(norm_expr, norm_ref.transpose(0, 1))  # (batch_size, n_cell_types)
        
        return torch.softmax(similarity * 5.0, dim=-1)  # Scale and apply softmax

def cell_wise_spearman_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Differentiable approximation of cell-wise Spearman correlation loss.
    This loss function prioritizes cell-wise correlation over gene-wise correlation.
    
    Args:
        predictions: Predicted gene expressions of shape (batch_size, n_genes)
        targets: Target gene expressions of shape (batch_size, n_genes)
        
    Returns:
        Loss value (1 - mean cell-wise Spearman correlation)
    """
    batch_size, n_genes = predictions.shape
    
    # Compute ranks for each cell (along gene dimension)
    def rank_tensor(x):
        # Sort values
        sorted_values, sorted_indices = torch.sort(x, dim=-1)
        # Get ranks
        ranks = torch.zeros_like(x)
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, n_genes)
        ranks[batch_indices, sorted_indices] = torch.arange(n_genes, device=x.device).float().expand(batch_size, -1)
        return ranks
    
    pred_ranks = rank_tensor(predictions)
    target_ranks = rank_tensor(targets)
    
    # Compute Spearman correlation for each cell
    pred_centered = pred_ranks - pred_ranks.mean(dim=-1, keepdim=True)
    target_centered = target_ranks - target_ranks.mean(dim=-1, keepdim=True)
    
    pred_var = torch.sum(pred_centered ** 2, dim=-1)
    target_var = torch.sum(target_centered ** 2, dim=-1)
    
    covariance = torch.sum(pred_centered * target_centered, dim=-1)
    correlation = covariance / (torch.sqrt(pred_var * target_var) + 1e-8)
    
    # Return loss (1 - correlation)
    return 1.0 - correlation.mean()

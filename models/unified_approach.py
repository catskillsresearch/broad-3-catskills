#!/usr/bin/env python3
# models/unified_approach.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer for modeling gene-gene relationships.
    Inspired by Tarandros' approach for improved gene expression prediction.
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Linear transformation
        self.W = nn.Linear(in_features, out_features, bias=False)
        # Attention mechanism
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        # Initialize with Xavier
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a.weight)
        
        # Leaky ReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h, adj):
        # Linear transformation
        Wh = self.W(h)  # (N, out_features)
        
        # Compute attention coefficients
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(self.a(a_input).squeeze(2))
        
        # Mask attention coefficients using adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime
    
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size(0)
        
        # Repeat to create all possible pairs
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        
        # Concatenate
        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        # Reshape to get a matrix where each row corresponds to a pair of nodes
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class SpatialAttention(nn.Module):
    """
    Spatial Attention module for integrating multi-level spatial context.
    Combines DeepSpot's multi-level approach with attention mechanisms.
    """
    def __init__(self, feature_dim, dropout=0.2):
        super(SpatialAttention, self).__init__()
        self.feature_dim = feature_dim
        
        # Query, Key, Value projections
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, spot_features, context_features, distances=None):
        """
        Apply spatial attention to integrate spot features with context features.
        
        Parameters:
        - spot_features: Features of the central spot (B, D)
        - context_features: Features of the context (neighbors, sub-spots) (B, N, D)
        - distances: Optional distances for distance-aware attention (B, N)
        
        Returns:
        - Integrated features (B, D)
        """
        # Reshape spot features for broadcasting
        q = self.query(spot_features).unsqueeze(1)  # (B, 1, D)
        
        # Project context features
        k = self.key(context_features)  # (B, N, D)
        v = self.value(context_features)  # (B, N, D)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.feature_dim ** 0.5)  # (B, 1, N)
        
        # Apply distance-based weighting if distances are provided
        if distances is not None:
            # Convert distances to weights (closer = higher weight)
            distance_weights = 1.0 / (distances.unsqueeze(1) + 1.0)  # (B, 1, N)
            scores = scores * distance_weights
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, N)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v).squeeze(1)  # (B, D)
        
        # Combine with original spot features
        output = self.out_proj(context + spot_features)  # (B, D)
        
        return output


class EnhancedCellEmbedding(nn.Module):
    """
    Enhanced Cell Embedding module that combines DeepSpot's multi-level features
    with Tarandros' cell-wise optimization approach.
    """
    def __init__(self, input_dim, embedding_dim, n_heads=4, dropout=0.3):
        super(EnhancedCellEmbedding, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Initial projection
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        self.bn1 = nn.BatchNorm1d(embedding_dim)
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(embedding_dim, n_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Initial projection
        x = F.relu(self.bn1(self.input_proj(x)))
        
        # Reshape for self-attention
        x_reshaped = x.unsqueeze(1)  # (B, 1, D)
        
        # Self-attention
        attn_output, _ = self.self_attn(x_reshaped, x_reshaped, x_reshaped)
        attn_output = attn_output.squeeze(1)  # (B, D)
        
        # Residual connection and normalization
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network
        ff_output = self.ff(x)
        
        # Residual connection and normalization
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class CellTypeEmbedding(nn.Module):
    """
    Cell type embedding module that helps capture cell-specific patterns.
    This is a key component of the Tarandros approach for cell-wise optimization.
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


class UnifiedDeepSpotModel(nn.Module):
    """
    Unified model that combines the best of DeepSpot and Tarandros approaches
    with sophisticated neural network architecture and comprehensive framework.
    """
    def __init__(
        self,
        config: Dict,
        use_gene_graph: bool = True
    ):
        super().__init__()
        # Extract configuration parameters
        self.feature_dim = config.get('feature_dim', 512)
        self.phi_dim = config.get('phi_dim', 256)
        self.embedding_dim = config.get('embedding_dim', 512)
        self.n_heads = config.get('n_heads', 4)
        self.dropout = config.get('dropout', 0.3)
        self.n_measured_genes = config.get('n_measured_genes', 460)
        self.n_unmeasured_genes = config.get('n_unmeasured_genes', 18157)
        self.n_cell_types = config.get('n_cell_types', 10)
        self.use_gene_graph = use_gene_graph
        
        # Initialize DeepSpot components
        self.phi_spot = nn.Sequential(
            nn.Linear(self.feature_dim, self.phi_dim * 2),
            nn.BatchNorm1d(self.phi_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.phi_dim * 2, self.phi_dim),
            nn.BatchNorm1d(self.phi_dim),
            nn.ReLU()
        )
        
        self.phi_subspot = nn.Sequential(
            nn.Linear(self.feature_dim, self.phi_dim * 2),
            nn.BatchNorm1d(self.phi_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.phi_dim * 2, self.phi_dim),
            nn.BatchNorm1d(self.phi_dim),
            nn.ReLU()
        )
        
        self.phi_neighbor = nn.Sequential(
            nn.Linear(self.feature_dim, self.phi_dim * 2),
            nn.BatchNorm1d(self.phi_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.phi_dim * 2, self.phi_dim),
            nn.BatchNorm1d(self.phi_dim),
            nn.ReLU()
        )
        
        # Spatial attention for integrating multi-level features
        self.subspot_attention = SpatialAttention(self.phi_dim, self.dropout)
        self.neighbor_attention = SpatialAttention(self.phi_dim, self.dropout)
        
        # Cell type embedding for cell-wise optimization
        self.cell_type_embedding = CellTypeEmbedding(
            self.phi_dim * 3,
            self.embedding_dim,
            n_cell_types=self.n_cell_types
        )
        
        # Enhanced cell embedding with transformer-like architecture
        self.cell_embedding = EnhancedCellEmbedding(
            self.embedding_dim + self.phi_dim * 3, 
            self.embedding_dim,
            n_heads=self.n_heads,
            dropout=self.dropout
        )
        
        # Gene-gene relationship modeling (optional)
        if self.use_gene_graph:
            self.gene_graph_layer = GraphAttentionLayer(
                self.embedding_dim, 
                self.embedding_dim,
                dropout=self.dropout
            )
            
            # Gene adjacency matrix (placeholder - would be initialized from real data)
            self.register_buffer(
                'gene_adj',
                torch.ones(self.n_measured_genes, self.n_measured_genes) - torch.eye(self.n_measured_genes)
            )
        
        # Prediction heads for measured and unmeasured genes
        self.measured_gene_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.BatchNorm1d(self.embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.n_measured_genes)
        )
        
        self.unmeasured_gene_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim + self.n_measured_genes, self.embedding_dim * 2),
            nn.BatchNorm1d(self.embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.n_unmeasured_genes)
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
    
    def forward(
        self,
        spot_features: torch.Tensor,
        subspot_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_distances: Optional[torch.Tensor] = None,
        predict_unmeasured: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the unified model.
        
        Parameters:
        - spot_features: Features of the central spot (B, D)
        - subspot_features: Features of sub-spots (B, N_sub, D)
        - neighbor_features: Features of neighboring spots (B, N_neigh, D)
        - neighbor_distances: Distances to neighboring spots (B, N_neigh)
        - predict_unmeasured: Whether to predict unmeasured genes
        
        Returns:
        - Dictionary with predictions for measured and unmeasured genes
        """
        batch_size = spot_features.shape[0]
        
        # Process spot features
        spot_embedding = self.phi_spot(spot_features)
        
        # Process sub-spot features
        # Reshape for batch processing
        subspot_flat = subspot_features.reshape(-1, subspot_features.shape[-1])
        subspot_processed = self.phi_subspot(subspot_flat)
        subspot_processed = subspot_processed.reshape(batch_size, -1, self.phi_dim)
        
        # Process neighbor features
        # Reshape for batch processing
        neighbor_flat = neighbor_features.reshape(-1, neighbor_features.shape[-1])
        neighbor_processed = self.phi_neighbor(neighbor_flat)
        neighbor_processed = neighbor_processed.reshape(batch_size, -1, self.phi_dim)
        
        # Apply spatial attention
        subspot_context = self.subspot_attention(spot_embedding, subspot_processed)
        neighbor_context = self.neighbor_attention(spot_embedding, neighbor_processed, neighbor_distances)
        
        # Concatenate features
        combined_features = torch.cat([spot_embedding, subspot_context, neighbor_context], dim=1)
        
        # Get cell type embeddings
        cell_type_logits, cell_type_embeddings = self.cell_type_embedding(combined_features)
        
        # Concatenate with combined features
        enhanced_features = torch.cat([combined_features, cell_type_embeddings], dim=1)
        
        # Apply enhanced cell embedding
        cell_embedding = self.cell_embedding(enhanced_features)
        
        # Predict measured genes
        if self.use_gene_graph:
            # First get initial gene predictions
            initial_predictions = self.measured_gene_predictor(cell_embedding)
            
            # Apply graph attention to refine predictions
            measured_predictions = initial_predictions + 0.1 * self.gene_graph_layer(
                initial_predictions, self.gene_adj
            )
        else:
            measured_predictions = self.measured_gene_predictor(cell_embedding)
        
        # Return early if we don't need to predict unmeasured genes
        if not predict_unmeasured:
            return {
                'measured_predictions': measured_predictions,
                'cell_embedding': cell_embedding,
                'cell_type_logits': cell_type_logits
            }
        
        # Predict unmeasured genes using both cell embedding and measured gene predictions
        unmeasured_input = torch.cat([cell_embedding, measured_predictions], dim=1)
        unmeasured_predictions = self.unmeasured_gene_predictor(unmeasured_input)
        
        # Apply similarity-based refinement using reference data if available
        if hasattr(self, 'reference_measured_expressions') and self.reference_measured_expressions.sum() > 0:
            # Compute similarity between measured expressions and reference measured expressions
            measured_similarity = self._compute_expression_similarity(
                measured_predictions, 
                self.reference_measured_expressions
            )  # (batch_size, n_cell_types)
            
            # Get cell type probabilities
            cell_type_probs = F.softmax(cell_type_logits, dim=-1)  # (batch_size, n_cell_types)
            
            # Combine cell type probabilities and expression similarity
            combined_weights = cell_type_probs * measured_similarity
            combined_weights = combined_weights / (combined_weights.sum(dim=-1, keepdim=True) + 1e-8)
            
            # Get reference-based predictions
            reference_predictions = torch.matmul(combined_weights, self.reference_expressions)
            
            # Blend model predictions with reference-based predictions
            alpha = 0.7  # Weight for model predictions vs. reference-based predictions
            unmeasured_predictions = alpha * unmeasured_predictions + (1 - alpha) * reference_predictions
        
        return {
            'measured_predictions': measured_predictions,
            'unmeasured_predictions': unmeasured_predictions,
            'cell_embedding': cell_embedding,
            'cell_type_logits': cell_type_logits
        }
    
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


def spearman_correlation_loss(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable approximation of Spearman correlation loss.
    
    Args:
        predictions: Predicted values (batch_size, n_features)
        targets: Target values (batch_size, n_features)
        eps: Small constant for numerical stability
        
    Returns:
        Loss value (1 - Spearman correlation)
    """
    # Convert to ranks
    pred_ranks = _to_ranks(predictions)
    target_ranks = _to_ranks(targets)
    
    # Compute mean
    pred_mean = pred_ranks.mean(dim=1, keepdim=True)
    target_mean = target_ranks.mean(dim=1, keepdim=True)
    
    # Compute covariance
    pred_diff = pred_ranks - pred_mean
    target_diff = target_ranks - target_mean
    cov = (pred_diff * target_diff).sum(dim=1)
    
    # Compute standard deviations
    pred_std = torch.sqrt((pred_diff ** 2).sum(dim=1) + eps)
    target_std = torch.sqrt((target_diff ** 2).sum(dim=1) + eps)
    
    # Compute correlation
    correlation = cov / (pred_std * target_std + eps)
    
    # Return loss (1 - correlation)
    return 1.0 - correlation.mean()


def cell_wise_spearman_loss(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable approximation of cell-wise Spearman correlation loss.
    This loss function prioritizes cell-wise correlation over gene-wise correlation.
    
    Args:
        predictions: Predicted gene expressions of shape (batch_size, n_genes)
        targets: Target gene expressions of shape (batch_size, n_genes)
        eps: Small constant for numerical stability
        
    Returns:
        Loss value (1 - mean cell-wise Spearman correlation)
    """
    # Compute ranks for each cell (along gene dimension)
    pred_ranks = _to_ranks(predictions)
    target_ranks = _to_ranks(targets)
    
    # Compute mean
    pred_mean = pred_ranks.mean(dim=1, keepdim=True)
    target_mean = target_ranks.mean(dim=1, keepdim=True)
    
    # Compute covariance
    pred_diff = pred_ranks - pred_mean
    target_diff = target_ranks - target_mean
    cov = (pred_diff * target_diff).sum(dim=1)
    
    # Compute standard deviations
    pred_std = torch.sqrt((pred_diff ** 2).sum(dim=1) + eps)
    target_std = torch.sqrt((target_diff ** 2).sum(dim=1) + eps)
    
    # Compute correlation
    correlation = cov / (pred_std * target_std + eps)
    
    # Return loss (1 - correlation)
    return 1.0 - correlation.mean()


def _to_ranks(x: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor values to ranks.
    
    Args:
        x: Input tensor of shape (batch_size, n_features)
        
    Returns:
        Tensor of same shape with values converted to ranks
    """
    batch_size, n_features = x.shape
    
    # Sort values
    sorted_values, sorted_indices = torch.sort(x, dim=1)
    
    # Create rank tensor
    ranks = torch.zeros_like(x)
    
    # Assign ranks
    for i in range(batch_size):
        ranks[i, sorted_indices[i]] = torch.arange(n_features, device=x.device, dtype=x.dtype)
    
    return ranks


class LogFCGeneRanking:
    """
    Log Fold Change (logFC) method for gene ranking.
    This is the approach used in Crunch 3 to identify genes that differentiate
    between dysplastic and non-dysplastic tissue regions.
    """
    def __init__(self, config: Dict):
        """
        Initialize the LogFC gene ranking method.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.gene_names = None
        self.gene_rankings = None
        self.logfc_values = None
        
    def compute_logfc(self, 
                     dysplastic_expressions: np.ndarray, 
                     non_dysplastic_expressions: np.ndarray,
                     gene_names: List[str]) -> np.ndarray:
        """
        Compute log fold change between dysplastic and non-dysplastic expressions.
        
        Args:
            dysplastic_expressions: Gene expressions for dysplastic cells (n_dysplastic_cells, n_genes)
            non_dysplastic_expressions: Gene expressions for non-dysplastic cells (n_non_dysplastic_cells, n_genes)
            gene_names: List of gene names
            
        Returns:
            Array of logFC values for each gene
        """
        # Store gene names
        self.gene_names = gene_names
        
        # Compute mean expression for each group
        dysplastic_mean = np.mean(dysplastic_expressions, axis=0)
        non_dysplastic_mean = np.mean(non_dysplastic_expressions, axis=0)
        
        # Add small constant to avoid division by zero or log(0)
        epsilon = 1e-10
        dysplastic_mean = dysplastic_mean + epsilon
        non_dysplastic_mean = non_dysplastic_mean + epsilon
        
        # Compute log fold change
        logfc = np.log2(dysplastic_mean / non_dysplastic_mean)
        
        # Store logFC values
        self.logfc_values = logfc
        
        return logfc
    
    def rank_genes(self, 
                  dysplastic_expressions: np.ndarray, 
                  non_dysplastic_expressions: np.ndarray,
                  gene_names: List[str]) -> pd.DataFrame:
        """
        Rank genes based on log fold change between dysplastic and non-dysplastic expressions.
        
        Args:
            dysplastic_expressions: Gene expressions for dysplastic cells (n_dysplastic_cells, n_genes)
            non_dysplastic_expressions: Gene expressions for non-dysplastic cells (n_non_dysplastic_cells, n_genes)
            gene_names: List of gene names
            
        Returns:
            DataFrame with gene rankings
        """
        import pandas as pd
        
        # Compute logFC
        logfc = self.compute_logfc(dysplastic_expressions, non_dysplastic_expressions, gene_names)
        
        # Create DataFrame
        gene_df = pd.DataFrame({
            'gene_name': gene_names,
            'logFC': logfc,
            'abs_logFC': np.abs(logfc)
        })
        
        # Rank genes by absolute logFC (higher absolute logFC = higher rank)
        gene_df = gene_df.sort_values('abs_logFC', ascending=False).reset_index(drop=True)
        gene_df['rank'] = gene_df.index + 1
        
        # Store gene rankings
        self.gene_rankings = gene_df
        
        return gene_df


class UnifiedPipeline:
    """
    Unified pipeline that combines the sophisticated neural network architecture
    from yesterday with the comprehensive framework from today.
    """
    def __init__(self, config: Dict):
        """
        Initialize the unified pipeline.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize unified model
        self.model = UnifiedDeepSpotModel(config)
        
        # Initialize logFC gene ranking for Crunch 3
        self.logfc_ranker = LogFCGeneRanking(config)
        
        # Paths for saving results
        self.output_dir = config.get('output_dir', 'output')
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_crunch1_and_2(self, 
                        spot_features: np.ndarray,
                        subspot_features: np.ndarray,
                        neighbor_features: np.ndarray,
                        neighbor_distances: np.ndarray,
                        measured_gene_expression: Optional[np.ndarray] = None,
                        reference_data: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """
        Run Crunch 1 and 2 to predict measured and unmeasured gene expression.
        
        Args:
            spot_features: Spot-level features (n_spots, feature_dim)
            subspot_features: Sub-spot features (n_spots, n_subspots, feature_dim)
            neighbor_features: Neighbor features (n_spots, n_neighbors, feature_dim)
            neighbor_distances: Neighbor distances (n_spots, n_neighbors)
            measured_gene_expression: Ground truth measured gene expression (n_spots, n_measured_genes)
            reference_data: Reference data for scRNA-seq
            
        Returns:
            Dictionary with predicted measured and unmeasured gene expression
        """
        print("Running Crunch 1 & 2: Predicting gene expression with unified model...")
        
        # Convert numpy arrays to torch tensors
        spot_features_tensor = torch.tensor(spot_features, dtype=torch.float32, device=self.device)
        subspot_features_tensor = torch.tensor(subspot_features, dtype=torch.float32, device=self.device)
        neighbor_features_tensor = torch.tensor(neighbor_features, dtype=torch.float32, device=self.device)
        neighbor_distances_tensor = torch.tensor(neighbor_distances, dtype=torch.float32, device=self.device)
        
        # Set reference data if provided
        if reference_data is not None:
            measured_ref = torch.tensor(reference_data['measured'], dtype=torch.float32, device=self.device)
            unmeasured_ref = torch.tensor(reference_data['unmeasured'], dtype=torch.float32, device=self.device)
            self.model.set_reference_data(measured_ref, unmeasured_ref)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Make predictions
        with torch.no_grad():
            predictions = self.model(
                spot_features_tensor,
                subspot_features_tensor,
                neighbor_features_tensor,
                neighbor_distances_tensor,
                predict_unmeasured=True
            )
        
        # Convert predictions to numpy
        measured_predictions_np = predictions['measured_predictions'].cpu().numpy()
        unmeasured_predictions_np = predictions['unmeasured_predictions'].cpu().numpy()
        
        # Evaluate if ground truth is provided
        if measured_gene_expression is not None:
            from scipy.stats import spearmanr
            
            # Compute gene-wise Spearman correlation
            gene_wise_corrs = []
            for i in range(measured_gene_expression.shape[1]):
                corr, _ = spearmanr(measured_predictions_np[:, i], measured_gene_expression[:, i])
                if not np.isnan(corr):
                    gene_wise_corrs.append(corr)
            
            mean_gene_wise_spearman = np.mean(gene_wise_corrs)
            print(f"Crunch 1 - Mean gene-wise Spearman correlation: {mean_gene_wise_spearman:.4f}")
            
            # Compute cell-wise Spearman correlation
            cell_wise_corrs = []
            for i in range(measured_gene_expression.shape[0]):
                corr, _ = spearmanr(measured_predictions_np[i], measured_gene_expression[i])
                if not np.isnan(corr):
                    cell_wise_corrs.append(corr)
            
            mean_cell_wise_spearman = np.mean(cell_wise_corrs)
            print(f"Crunch 1 - Mean cell-wise Spearman correlation: {mean_cell_wise_spearman:.4f}")
        
        # Save predictions
        np.save(os.path.join(self.output_dir, 'measured_predictions.npy'), measured_predictions_np)
        np.save(os.path.join(self.output_dir, 'unmeasured_predictions.npy'), unmeasured_predictions_np)
        
        return {
            'measured_predictions': measured_predictions_np,
            'unmeasured_predictions': unmeasured_predictions_np
        }
    
    def run_crunch3(self, 
                   gene_expression: np.ndarray,
                   cell_labels: np.ndarray,
                   gene_names: List[str]) -> pd.DataFrame:
        """
        Run Crunch 3 to rank genes using logFC method.
        
        Args:
            gene_expression: Gene expression (n_cells, n_genes)
            cell_labels: Cell labels (n_cells), 1 for dysplastic, 0 for non-dysplastic
            gene_names: List of gene names
            
        Returns:
            DataFrame with gene rankings
        """
        print("Running Crunch 3: Ranking genes with logFC method...")
        
        # Split expression data by cell type
        dysplastic_mask = cell_labels == 1
        non_dysplastic_mask = cell_labels == 0
        
        dysplastic_expression = gene_expression[dysplastic_mask]
        non_dysplastic_expression = gene_expression[non_dysplastic_mask]
        
        print(f"Number of dysplastic cells: {dysplastic_expression.shape[0]}")
        print(f"Number of non-dysplastic cells: {non_dysplastic_expression.shape[0]}")
        
        # Rank genes
        gene_rankings = self.logfc_ranker.rank_genes(
            dysplastic_expression,
            non_dysplastic_expression,
            gene_names
        )
        
        # Save rankings
        import os
        rankings_path = os.path.join(self.output_dir, 'gene_rankings.csv')
        os.makedirs(os.path.dirname(rankings_path), exist_ok=True)
        gene_rankings.to_csv(rankings_path, index=False)
        
        return gene_rankings
    
    def run_full_pipeline(self, 
                         spot_features: np.ndarray,
                         subspot_features: np.ndarray,
                         neighbor_features: np.ndarray,
                         neighbor_distances: np.ndarray,
                         measured_gene_expression: Optional[np.ndarray] = None,
                         cell_labels: Optional[np.ndarray] = None,
                         measured_gene_names: Optional[List[str]] = None,
                         unmeasured_gene_names: Optional[List[str]] = None,
                         reference_data: Optional[Dict] = None) -> Dict:
        """
        Run the full unified pipeline.
        
        Args:
            spot_features: Spot-level features (n_spots, feature_dim)
            subspot_features: Sub-spot features (n_spots, n_subspots, feature_dim)
            neighbor_features: Neighbor features (n_spots, n_neighbors, feature_dim)
            neighbor_distances: Neighbor distances (n_spots, n_neighbors)
            measured_gene_expression: Measured gene expression (n_spots, n_measured_genes)
            cell_labels: Cell labels (n_spots), 1 for dysplastic, 0 for non-dysplastic
            measured_gene_names: List of measured gene names
            unmeasured_gene_names: List of unmeasured gene names
            reference_data: Reference data for scRNA-seq
            
        Returns:
            Dictionary with results from all three crunches
        """
        print("Running full unified pipeline...")
        
        # Run Crunch 1 & 2
        predictions = self.run_crunch1_and_2(
            spot_features,
            subspot_features,
            neighbor_features,
            neighbor_distances,
            measured_gene_expression,
            reference_data
        )
        
        measured_predictions = predictions['measured_predictions']
        unmeasured_predictions = predictions['unmeasured_predictions']
        
        # Use predicted expression if ground truth is not available
        if measured_gene_expression is None:
            measured_gene_expression = measured_predictions
        
        # Combine measured and unmeasured gene expression
        all_gene_expression = np.concatenate([measured_gene_expression, unmeasured_predictions], axis=1)
        all_gene_names = measured_gene_names + unmeasured_gene_names if measured_gene_names and unmeasured_gene_names else None
        
        # Run Crunch 3 if cell labels are provided
        gene_rankings = None
        if cell_labels is not None and all_gene_names is not None:
            gene_rankings = self.run_crunch3(
                all_gene_expression,
                cell_labels,
                all_gene_names
            )
        
        # Return results
        results = {
            'measured_predictions': measured_predictions,
            'unmeasured_predictions': unmeasured_predictions,
            'combined_expression': all_gene_expression,
            'gene_rankings': gene_rankings
        }
        
        return results


import pytorch_lightning as pl

class UnifiedLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for the unified model.
    """
    def __init__(self, config: Dict):
        """
        Initialize the unified lightning module.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Initialize unified model
        self.model = UnifiedDeepSpotModel(config)
        
        # Loss weights
        self.gene_wise_weight = config.get('gene_wise_weight', 0.3)
        self.cell_wise_weight = config.get('cell_wise_weight', 0.7)
        
    def forward(
        self,
        spot_features: torch.Tensor,
        subspot_features: torch.Tensor,
        neighbor_features: torch.Tensor,
        neighbor_distances: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the unified model.
        
        Args:
            spot_features: Spot-level features (batch_size, feature_dim)
            subspot_features: Sub-spot features (batch_size, n_subspots, feature_dim)
            neighbor_features: Neighbor features (batch_size, n_neighbors, feature_dim)
            neighbor_distances: Neighbor distances (batch_size, n_neighbors)
            
        Returns:
            Dictionary with predictions
        """
        return self.model(
            spot_features,
            subspot_features,
            neighbor_features,
            neighbor_distances
        )
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
            
        Returns:
            Loss value
        """
        # Extract batch data
        spot_features = batch['spot_features']
        subspot_features = batch['subspot_features']
        neighbor_features = batch['neighbor_features']
        neighbor_distances = batch['neighbor_distances']
        measured_expression = batch['measured_expression']
        unmeasured_expression = batch.get('unmeasured_expression', None)
        
        # Forward pass
        predictions = self(
            spot_features,
            subspot_features,
            neighbor_features,
            neighbor_distances
        )
        
        measured_predictions = predictions['measured_predictions']
        
        # Compute loss for measured genes
        gene_wise_loss = spearman_correlation_loss(measured_predictions, measured_expression)
        cell_wise_loss = cell_wise_spearman_loss(measured_predictions, measured_expression)
        
        # Combine losses with weights
        measured_loss = self.gene_wise_weight * gene_wise_loss + self.cell_wise_weight * cell_wise_loss
        
        # Compute loss for unmeasured genes if available
        unmeasured_loss = 0.0
        if unmeasured_expression is not None and 'unmeasured_predictions' in predictions:
            unmeasured_predictions = predictions['unmeasured_predictions']
            unmeasured_gene_wise_loss = spearman_correlation_loss(unmeasured_predictions, unmeasured_expression)
            unmeasured_cell_wise_loss = cell_wise_spearman_loss(unmeasured_predictions, unmeasured_expression)
            unmeasured_loss = self.gene_wise_weight * unmeasured_gene_wise_loss + self.cell_wise_weight * unmeasured_cell_wise_loss
        
        # Total loss
        total_loss = measured_loss + unmeasured_loss
        
        # Log metrics
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_measured_loss', measured_loss, on_step=True, on_epoch=True)
        self.log('train_gene_wise_loss', gene_wise_loss, on_step=True, on_epoch=True)
        self.log('train_cell_wise_loss', cell_wise_loss, on_step=True, on_epoch=True)
        
        if unmeasured_expression is not None and 'unmeasured_predictions' in predictions:
            self.log('train_unmeasured_loss', unmeasured_loss, on_step=True, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Validation step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
        """
        # Extract batch data
        spot_features = batch['spot_features']
        subspot_features = batch['subspot_features']
        neighbor_features = batch['neighbor_features']
        neighbor_distances = batch['neighbor_distances']
        measured_expression = batch['measured_expression']
        unmeasured_expression = batch.get('unmeasured_expression', None)
        
        # Forward pass
        predictions = self(
            spot_features,
            subspot_features,
            neighbor_features,
            neighbor_distances
        )
        
        measured_predictions = predictions['measured_predictions']
        
        # Compute loss for measured genes
        gene_wise_loss = spearman_correlation_loss(measured_predictions, measured_expression)
        cell_wise_loss = cell_wise_spearman_loss(measured_predictions, measured_expression)
        
        # Combine losses with weights
        measured_loss = self.gene_wise_weight * gene_wise_loss + self.cell_wise_weight * cell_wise_loss
        
        # Compute loss for unmeasured genes if available
        unmeasured_loss = 0.0
        if unmeasured_expression is not None and 'unmeasured_predictions' in predictions:
            unmeasured_predictions = predictions['unmeasured_predictions']
            unmeasured_gene_wise_loss = spearman_correlation_loss(unmeasured_predictions, unmeasured_expression)
            unmeasured_cell_wise_loss = cell_wise_spearman_loss(unmeasured_predictions, unmeasured_expression)
            unmeasured_loss = self.gene_wise_weight * unmeasured_gene_wise_loss + self.cell_wise_weight * unmeasured_cell_wise_loss
        
        # Total loss
        total_loss = measured_loss + unmeasured_loss
        
        # Log metrics
        self.log('val_loss', total_loss, on_epoch=True, prog_bar=True)
        self.log('val_measured_loss', measured_loss, on_epoch=True)
        self.log('val_gene_wise_loss', gene_wise_loss, on_epoch=True)
        self.log('val_cell_wise_loss', cell_wise_loss, on_epoch=True)
        
        if unmeasured_expression is not None and 'unmeasured_predictions' in predictions:
            self.log('val_unmeasured_loss', unmeasured_loss, on_epoch=True)
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """
        Test step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
        """
        # Extract batch data
        spot_features = batch['spot_features']
        subspot_features = batch['subspot_features']
        neighbor_features = batch['neighbor_features']
        neighbor_distances = batch['neighbor_distances']
        measured_expression = batch['measured_expression']
        unmeasured_expression = batch.get('unmeasured_expression', None)
        
        # Forward pass
        predictions = self(
            spot_features,
            subspot_features,
            neighbor_features,
            neighbor_distances
        )
        
        measured_predictions = predictions['measured_predictions']
        
        # Compute metrics for measured genes
        gene_wise_loss = spearman_correlation_loss(measured_predictions, measured_expression)
        cell_wise_loss = cell_wise_spearman_loss(measured_predictions, measured_expression)
        
        # Log metrics
        self.log('test_gene_wise_loss', gene_wise_loss, on_epoch=True)
        self.log('test_cell_wise_loss', cell_wise_loss, on_epoch=True)
        
        # Compute metrics for unmeasured genes if available
        if unmeasured_expression is not None and 'unmeasured_predictions' in predictions:
            unmeasured_predictions = predictions['unmeasured_predictions']
            unmeasured_gene_wise_loss = spearman_correlation_loss(unmeasured_predictions, unmeasured_expression)
            unmeasured_cell_wise_loss = cell_wise_spearman_loss(unmeasured_predictions, unmeasured_expression)
            
            self.log('test_unmeasured_gene_wise_loss', unmeasured_gene_wise_loss, on_epoch=True)
            self.log('test_unmeasured_cell_wise_loss', unmeasured_cell_wise_loss, on_epoch=True)
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Predict step.
        
        Args:
            batch: Batch dictionary
            batch_idx: Batch index
            
        Returns:
            Dictionary with predictions
        """
        # Extract batch data
        spot_features = batch['spot_features']
        subspot_features = batch['subspot_features']
        neighbor_features = batch['neighbor_features']
        neighbor_distances = batch['neighbor_distances']
        
        # Forward pass
        predictions = self(
            spot_features,
            subspot_features,
            neighbor_features,
            neighbor_distances
        )
        
        return predictions
    
    def configure_optimizers(self):
        """
        Configure optimizers.
        
        Returns:
            Optimizer
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 0.0001)
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }

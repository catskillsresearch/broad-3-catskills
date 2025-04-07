import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union

class GraphAttentionLayer(nn.Module):
    """Graph attention layer for modeling gene-gene relationships."""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3, alpha: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the graph attention layer.
        
        Args:
            h: Node features of shape [batch_size, num_nodes, in_features]
            adj: Adjacency matrix of shape [batch_size, num_nodes, num_nodes]
            
        Returns:
            Updated node features of shape [batch_size, num_nodes, out_features]
        """
        batch_size, N, _ = h.size()
        
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # [batch_size, N, out_features]
        
        # Attention mechanism
        a_input = torch.cat([Wh.repeat(1, 1, N).view(batch_size, N * N, self.out_features),
                             Wh.repeat(1, N, 1)], dim=2).view(batch_size, N, N, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(3))  # [batch_size, N, N]
        
        # Mask attention coefficients using adjacency matrix
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)  # [batch_size, N, out_features]
        
        return h_prime

class SpatialAttention(nn.Module):
    """Spatial attention mechanism for integrating multi-level spatial context."""
    
    def __init__(self, feature_dim: int, dropout: float = 0.3):
        super(SpatialAttention, self).__init__()
        self.feature_dim = feature_dim
        self.dropout = dropout
        
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, query: torch.Tensor, context: torch.Tensor, 
                distances: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the spatial attention mechanism.
        
        Args:
            query: Query features of shape [batch_size, 1, feature_dim]
            context: Context features of shape [batch_size, num_context, feature_dim]
            distances: Optional distances between query and context points of shape [batch_size, num_context]
            
        Returns:
            Attended context features of shape [batch_size, feature_dim]
        """
        # Project query, keys, and values
        q = self.query_proj(query)  # [batch_size, 1, feature_dim]
        k = self.key_proj(context)  # [batch_size, num_context, feature_dim]
        v = self.value_proj(context)  # [batch_size, num_context, feature_dim]
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.feature_dim ** 0.5)  # [batch_size, 1, num_context]
        
        # Apply distance weighting if provided
        if distances is not None:
            # Convert distances to weights (closer points get higher weights)
            distance_weights = 1.0 / (distances.unsqueeze(1) + 1.0)
            scores = scores * distance_weights
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, 1, num_context]
        attn_weights = self.dropout_layer(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v).squeeze(1)  # [batch_size, feature_dim]
        
        return context

class EnhancedCellEmbedding(nn.Module):
    """Transformer-like architecture with multi-head attention for cell embedding."""
    
    def __init__(self, input_dim: int, embedding_dim: int, num_heads: int = 4, dropout: float = 0.3):
        super(EnhancedCellEmbedding, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the enhanced cell embedding module.
        
        Args:
            x: Input features of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Enhanced cell embeddings of shape [batch_size, seq_len, embedding_dim]
        """
        # Project input to embedding dimension
        x = self.input_proj(x)  # [batch_size, seq_len, embedding_dim]
        
        # Multi-head self-attention
        attn_output, _ = self.multihead_attn(x, x, x)
        x = x + self.dropout_layer(attn_output)
        x = self.norm1(x)
        
        # Feed-forward network
        ffn_output = self.ffn(x)
        x = x + self.dropout_layer(ffn_output)
        x = self.norm2(x)
        
        return x

class CellTypeEmbedding(nn.Module):
    """Cell type prediction and embedding for cell-specific patterns."""
    
    def __init__(self, input_dim: int, embedding_dim: int, num_cell_types: int, dropout: float = 0.3):
        super(CellTypeEmbedding, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_cell_types = num_cell_types
        self.dropout = dropout
        
        # Cell type classifier
        self.cell_type_classifier = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, num_cell_types)
        )
        
        # Cell type embeddings
        self.cell_type_embeddings = nn.Parameter(
            torch.randn(num_cell_types, embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the cell type embedding module.
        
        Args:
            x: Input features of shape [batch_size, input_dim]
            
        Returns:
            Tuple containing:
                - Cell type logits of shape [batch_size, num_cell_types]
                - Cell type probabilities of shape [batch_size, num_cell_types]
                - Cell type embeddings of shape [batch_size, embedding_dim]
        """
        # Predict cell types
        cell_type_logits = self.cell_type_classifier(x)  # [batch_size, num_cell_types]
        cell_type_probs = F.softmax(cell_type_logits, dim=-1)  # [batch_size, num_cell_types]
        
        # Get cell type embeddings
        cell_embedding = torch.matmul(cell_type_probs, self.cell_type_embeddings)  # [batch_size, embedding_dim]
        
        return cell_type_logits, cell_type_probs, cell_embedding

class UnifiedDeepSpotModel(nn.Module):
    """
    Unified model that combines DeepSpot and Tarandros approaches for spatial transcriptomics analysis.
    
    This model integrates:
    1. DeepSpot's multi-level spatial context and deep-set architecture
    2. Tarandros's cell-wise optimization and reference dataset integration
    3. Sophisticated attention mechanisms and graph neural networks
    """
    
    def __init__(self, config: Dict):
        super(UnifiedDeepSpotModel, self).__init__()
        
        # Configuration parameters
        self.feature_dim = config.get('feature_dim', 512)
        self.phi_dim = config.get('phi_dim', 256)
        self.embedding_dim = config.get('embedding_dim', 512)
        self.n_heads = config.get('n_heads', 4)
        self.dropout = config.get('dropout', 0.3)
        self.n_measured_genes = config.get('n_measured_genes', 460)
        self.n_unmeasured_genes = config.get('n_unmeasured_genes', 18157)
        self.n_cell_types = config.get('n_cell_types', 10)
        self.use_gene_graph = config.get('use_gene_graph', True)
        self.use_reference_data = config.get('use_reference_data', True)
        self.aggregation_mode = config.get('aggregation_mode', 'attention')  # Options: sum, max, mean, attention
        
        # Initialize gene adjacency matrix if using gene graph
        if self.use_gene_graph:
            self.gene_adj = nn.Parameter(torch.ones(self.n_measured_genes, self.n_measured_genes))
        
        # Initialize reference data if using reference integration
        if self.use_reference_data:
            self.register_buffer('reference_measured_expressions', 
                                torch.randn(100, self.n_measured_genes))  # Placeholder
            self.register_buffer('reference_expressions', 
                                torch.randn(100, self.n_unmeasured_genes))  # Placeholder
        
        # Feature processing networks (phi networks)
        self.phi_spot = nn.Sequential(
            nn.Linear(self.feature_dim, self.phi_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.phi_dim, self.phi_dim)
        )
        
        self.phi_subspot = nn.Sequential(
            nn.Linear(self.feature_dim, self.phi_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.phi_dim, self.phi_dim)
        )
        
        self.phi_neighbor = nn.Sequential(
            nn.Linear(self.feature_dim, self.phi_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.phi_dim, self.phi_dim)
        )
        
        # Spatial attention mechanisms
        self.subspot_attention = SpatialAttention(self.phi_dim, self.dropout)
        self.neighbor_attention = SpatialAttention(self.phi_dim, self.dropout)
        
        # Cell embedding and cell type prediction
        self.cell_embedding = EnhancedCellEmbedding(
            input_dim=self.phi_dim * 3,  # Concatenated spot, subspot, and neighbor features
            embedding_dim=self.embedding_dim,
            num_heads=self.n_heads,
            dropout=self.dropout
        )
        
        self.cell_type_embedding = CellTypeEmbedding(
            input_dim=self.embedding_dim,
            embedding_dim=self.embedding_dim,
            num_cell_types=self.n_cell_types,
            dropout=self.dropout
        )
        
        # Gene prediction networks
        self.measured_gene_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.n_measured_genes)
        )
        
        self.unmeasured_gene_predictor = nn.Sequential(
            nn.Linear(self.embedding_dim + self.n_measured_genes, self.embedding_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.n_unmeasured_genes)
        )
        
        # Graph attention for gene-gene relationships
        if self.use_gene_graph:
            self.gene_graph_layer = GraphAttentionLayer(
                in_features=self.n_measured_genes,
                out_features=self.n_measured_genes,
                dropout=self.dropout
            )
    
    def _compute_expression_similarity(self, pred_expressions: torch.Tensor, 
                                      ref_expressions: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity between predicted expressions and reference expressions.
        
        Args:
            pred_expressions: Predicted expressions of shape [batch_size, n_genes]
            ref_expressions: Reference expressions of shape [n_ref_cells, n_genes]
            
        Returns:
            Similarity matrix of shape [batch_size, n_ref_cells]
        """
        # Normalize expressions
        pred_norm = F.normalize(pred_expressions, p=2, dim=1)
        ref_norm = F.normalize(ref_expressions, p=2, dim=1)
        
        # Calculate cosine similarity
        similarity = torch.matmul(pred_norm, ref_norm.transpose(0, 1))
        
        return similarity
    
    def forward(self, spot_features: torch.Tensor, subspot_features: torch.Tensor, 
                neighbor_features: torch.Tensor, subspot_distances: Optional[torch.Tensor] = None,
                neighbor_distances: Optional[torch.Tensor] = None,
                spatial_coordinates: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the unified model.
        
        Args:
            spot_features: Spot-level features of shape [batch_size, feature_dim]
            subspot_features: Sub-spot features of shape [batch_size, n_subspots, feature_dim]
            neighbor_features: Neighbor features of shape [batch_size, n_neighbors, feature_dim]
            subspot_distances: Optional distances to sub-spots of shape [batch_size, n_subspots]
            neighbor_distances: Optional distances to neighbors of shape [batch_size, n_neighbors]
            spatial_coordinates: Optional spatial coordinates of shape [batch_size, 2]
            
        Returns:
            Dictionary containing:
                - measured_predictions: Predicted measured gene expressions
                - unmeasured_predictions: Predicted unmeasured gene expressions
                - cell_type_logits: Cell type logits
                - cell_type_probs: Cell type probabilities
                - cell_embedding: Cell embedding
        """
        batch_size = spot_features.size(0)
        
        # Process spot features
        spot_phi = self.phi_spot(spot_features)  # [batch_size, phi_dim]
        
        # Process sub-spot features
        subspot_phi = self.phi_subspot(subspot_features)  # [batch_size, n_subspots, phi_dim]
        
        # Process neighbor features
        neighbor_phi = self.phi_neighbor(neighbor_features)  # [batch_size, n_neighbors, phi_dim]
        
        # Integrate sub-spot context using attention
        subspot_context = self.subspot_attention(
            query=spot_phi.unsqueeze(1),
            context=subspot_phi,
            distances=subspot_distances
        )  # [batch_size, phi_dim]
        
        # Integrate neighbor context using attention
        neighbor_context = self.neighbor_attention(
            query=spot_phi.unsqueeze(1),
            context=neighbor_phi,
            distances=neighbor_distances
        )  # [batch_size, phi_dim]
        
        # Concatenate all features
        combined_features = torch.cat([
            spot_phi, subspot_context, neighbor_context
        ], dim=1)  # [batch_size, phi_dim * 3]
        
        # Generate cell embedding
        cell_embedding = self.cell_embedding(combined_features.unsqueeze(1)).squeeze(1)  # [batch_size, embedding_dim]
        
        # Predict cell types and get cell type embedding
        cell_type_logits, cell_type_probs, cell_type_embedding = self.cell_type_embedding(cell_embedding)
        
        # Enhance cell embedding with cell type information
        enhanced_cell_embedding = cell_embedding + 0.1 * cell_type_embedding
        
        # Predict measured genes
        if self.use_gene_graph:
            # Apply graph attention to refine predictions
            initial_predictions = self.measured_gene_predictor(enhanced_cell_embedding)
            measured_predictions = initial_predictions + 0.1 * self.gene_graph_layer(
                initial_predictions.unsqueeze(1),
                self.gene_adj.expand(batch_size, -1, -1)
            ).squeeze(1)
        else:
            measured_predictions = self.measured_gene_predictor(enhanced_cell_embedding)
        
        # Predict unmeasured genes
        combined_input = torch.cat([enhanced_cell_embedding, measured_predictions], dim=1)
        unmeasured_predictions = self.unmeasured_gene_predictor(combined_input)
        
        # Apply reference-based refinement if enabled
        if self.use_reference_data:
            # Calculate similarity between predicted measured expressions and reference measured expressions
            measured_similarity = self._compute_expression_similarity(
                measured_predictions, 
                self.reference_measured_expressions
            )
            
            # Combine cell type probabilities and expression similarity
            combined_weights = cell_type_probs.unsqueeze(1) * measured_similarity.unsqueeze(2)
            combined_weights = combined_weights.sum(dim=2)  # [batch_size, n_ref_cells]
            combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)
            
            # Get reference-based predictions
            reference_predictions = torch.matmul(combined_weights, self.reference_expressions)
            
            # Blend model predictions with reference predictions
            alpha = 0.7  # Weight for model predictions vs. reference predictions
            unmeasured_predictions = alpha * unmeasured_predictions + (1 - alpha) * reference_predictions
        
        return {
            'measured_predictions': measured_predictions,
            'unmeasured_predictions': unmeasured_predictions,
            'cell_type_logits': cell_type_logits,
            'cell_type_probs': cell_type_probs,
            'cell_embedding': enhanced_cell_embedding
        }

class LogFCGeneRanking:
    """
    Implementation of the logFC method for gene ranking.
    
    This class provides functionality to:
    1. Calculate log fold change between dysplastic and non-dysplastic regions
    2. Perform statistical testing to assess significance
    3. Rank genes based on their ability to differentiate between tissue types
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the LogFCGeneRanking class.
        
        Args:
            config: Configuration dictionary with parameters for gene ranking
        """
        self.config = config
        self.epsilon = config.get('epsilon', 1e-10)
        self.fdr_threshold = config.get('fdr_threshold', 0.05)
        self.logfc_threshold = config.get('logfc_threshold', 1.0)
    
    def calculate_logfc(self, gene_expression: np.ndarray, cell_labels: np.ndarray) -> Dict:
        """
        Calculate log fold change between dysplastic and non-dysplastic regions.
        
        Args:
            gene_expression: Gene expression matrix of shape [n_cells, n_genes]
            cell_labels: Cell labels of shape [n_cells], with values 'dysplastic' or 'non_dysplastic'
            
        Returns:
            Dictionary containing:
                - log_fold_change: Log fold change values for each gene
                - p_values: P-values from statistical testing
                - q_values: FDR-adjusted p-values
                - significant_genes: Indices of significant genes
                - ranked_genes: Indices of genes ranked by absolute log fold change
        """
        # Define groups based on cell labels
        dysplastic_mask = cell_labels == 'dysplastic'
        non_dysplastic_mask = cell_labels == 'non_dysplastic'
        
        # Get expression data for each group
        dysplastic_expr = gene_expression[dysplastic_mask]
        non_dysplastic_expr = gene_expression[non_dysplastic_mask]
        
        # Aggregate expression within each group (using median for robustness)
        dysplastic_mean = np.median(dysplastic_expr, axis=0)
        non_dysplastic_mean = np.median(non_dysplastic_expr, axis=0)
        
        # Calculate log fold change
        log_fold_change = np.log2((dysplastic_mean + self.epsilon) / (non_dysplastic_mean + self.epsilon))
        
        # Perform statistical testing (Wilcoxon rank-sum test for robustness)
        p_values = []
        for i in range(gene_expression.shape[1]):
            try:
                from scipy import stats
                _, p_val = stats.ranksums(
                    dysplastic_expr[:, i],
                    non_dysplastic_expr[:, i]
                )
                p_values.append(p_val)
            except:
                # Fallback to t-test if scipy is not available
                t_stat, p_val = self._ttest_ind(
                    dysplastic_expr[:, i],
                    non_dysplastic_expr[:, i]
                )
                p_values.append(p_val)
        p_values = np.array(p_values)
        
        # Apply multiple testing correction
        q_values = self._adjust_pvalues(p_values)
        
        # Identify significant genes
        significant_genes = np.where((q_values < self.fdr_threshold) & 
                                    (np.abs(log_fold_change) > self.logfc_threshold))[0]
        
        # Rank genes by absolute log fold change among significant genes
        ranked_genes = significant_genes[np.argsort(np.abs(log_fold_change[significant_genes]))[::-1]]
        
        return {
            'log_fold_change': log_fold_change,
            'p_values': p_values,
            'q_values': q_values,
            'significant_genes': significant_genes,
            'ranked_genes': ranked_genes
        }
    
    def _ttest_ind(self, a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        """
        Simple implementation of Welch's t-test for unequal variances.
        
        Args:
            a: First sample
            b: Second sample
            
        Returns:
            Tuple containing:
                - t-statistic
                - p-value
        """
        # Calculate means
        mean_a = np.mean(a)
        mean_b = np.mean(b)
        
        # Calculate variances
        var_a = np.var(a, ddof=1)
        var_b = np.var(b, ddof=1)
        
        # Calculate standard error
        n_a = len(a)
        n_b = len(b)
        se = np.sqrt(var_a / n_a + var_b / n_b)
        
        # Calculate t-statistic
        t_stat = (mean_a - mean_b) / se
        
        # Calculate degrees of freedom (Welch-Satterthwaite equation)
        df = (var_a / n_a + var_b / n_b) ** 2 / ((var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1))
        
        # Calculate p-value (approximation)
        p_val = 2 * (1 - self._t_cdf(np.abs(t_stat), df))
        
        return t_stat, p_val
    
    def _t_cdf(self, t: float, df: float) -> float:
        """
        Simple approximation of the CDF of the t-distribution.
        
        Args:
            t: t-statistic
            df: Degrees of freedom
            
        Returns:
            Cumulative probability
        """
        # Use normal approximation for large df
        if df > 100:
            from math import erf, sqrt
            return 0.5 * (1 + erf(t / sqrt(2)))
        
        # Simple approximation for smaller df
        x = df / (df + t * t)
        return 1 - 0.5 * x ** (df / 2)
    
    def _adjust_pvalues(self, p_values: np.ndarray) -> np.ndarray:
        """
        Apply Benjamini-Hochberg FDR correction to p-values.
        
        Args:
            p_values: Array of p-values
            
        Returns:
            Array of adjusted p-values (q-values)
        """
        n = len(p_values)
        
        # Sort p-values
        sorted_indices = np.argsort(p_values)
        sorted_pvalues = p_values[sorted_indices]
        
        # Calculate adjusted p-values
        adjusted_pvalues = np.zeros_like(sorted_pvalues)
        adjusted_pvalues[-1] = sorted_pvalues[-1]
        for i in range(n-2, -1, -1):
            adjusted_pvalues[i] = min(sorted_pvalues[i] * n / (i + 1), adjusted_pvalues[i+1])
        
        # Restore original order
        original_order = np.zeros_like(sorted_indices)
        original_order[sorted_indices] = np.arange(n)
        q_values = adjusted_pvalues[original_order]
        
        return q_values
    
    def create_volcano_plot(self, log_fold_change: np.ndarray, p_values: np.ndarray, 
                           gene_names: List[str], output_path: str) -> None:
        """
        Create a volcano plot of differential expression results.
        
        Args:
            log_fold_change: Log fold change values for each gene
            p_values: P-values from statistical testing
            gene_names: List of gene names
            output_path: Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create figure
            plt.figure(figsize=(10, 8))
            
            # Calculate -log10(p-value)
            neg_log_p = -np.log10(p_values + self.epsilon)
            
            # Define significant genes
            significant = (p_values < self.fdr_threshold) & (np.abs(log_fold_change) > self.logfc_threshold)
            
            # Plot non-significant genes
            plt.scatter(
                log_fold_change[~significant],
                neg_log_p[~significant],
                alpha=0.5,
                color='gray',
                label='Not significant'
            )
            
            # Plot significant genes
            plt.scatter(
                log_fold_change[significant],
                neg_log_p[significant],
                alpha=0.8,
                color='red',
                label='Significant'
            )
            
            # Label top genes
            top_genes = np.argsort(neg_log_p * np.abs(log_fold_change))[-10:]
            for i in top_genes:
                plt.annotate(
                    gene_names[i],
                    (log_fold_change[i], neg_log_p[i]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )
            
            # Add threshold lines
            plt.axhline(-np.log10(self.fdr_threshold), color='red', linestyle='--')
            plt.axvline(-self.logfc_threshold, color='blue', linestyle='--')
            plt.axvline(self.logfc_threshold, color='blue', linestyle='--')
            
            # Add labels and title
            plt.xlabel('Log2 Fold Change')
            plt.ylabel('-Log10 P-value')
            plt.title('Volcano Plot of Differential Expression')
            plt.legend()
            
            # Save figure
            plt.savefig(output_path)
            plt.close()
        except:
            print("Failed to create volcano plot. Matplotlib may not be available.")

class UnifiedPipeline:
    """
    Comprehensive pipeline that integrates DeepSpot, Tarandros, and LogFC approaches.
    
    This pipeline provides a complete workflow for:
    1. Predicting measured gene expression (DeepSpot)
    2. Inferring unmeasured gene expression (Tarandros)
    3. Ranking genes based on their ability to differentiate tissue types (LogFC)
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the UnifiedPipeline.
        
        Args:
            config: Configuration dictionary with parameters for the pipeline
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = UnifiedDeepSpotModel(config).to(self.device)
        
        # Initialize LogFC gene ranking
        self.gene_ranking = LogFCGeneRanking(config)
    
    def load_model(self, checkpoint_path: str) -> None:
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {checkpoint_path}")
    
    def predict(self, spot_features: np.ndarray, subspot_features: np.ndarray, 
               neighbor_features: np.ndarray, subspot_distances: Optional[np.ndarray] = None,
               neighbor_distances: Optional[np.ndarray] = None,
               spatial_coordinates: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Run prediction using the unified model.
        
        Args:
            spot_features: Spot-level features of shape [batch_size, feature_dim]
            subspot_features: Sub-spot features of shape [batch_size, n_subspots, feature_dim]
            neighbor_features: Neighbor features of shape [batch_size, n_neighbors, feature_dim]
            subspot_distances: Optional distances to sub-spots of shape [batch_size, n_subspots]
            neighbor_distances: Optional distances to neighbors of shape [batch_size, n_neighbors]
            spatial_coordinates: Optional spatial coordinates of shape [batch_size, 2]
            
        Returns:
            Dictionary containing:
                - measured_predictions: Predicted measured gene expressions
                - unmeasured_predictions: Predicted unmeasured gene expressions
                - cell_type_probs: Cell type probabilities
        """
        # Convert inputs to tensors
        spot_features = torch.tensor(spot_features, dtype=torch.float32).to(self.device)
        subspot_features = torch.tensor(subspot_features, dtype=torch.float32).to(self.device)
        neighbor_features = torch.tensor(neighbor_features, dtype=torch.float32).to(self.device)
        
        if subspot_distances is not None:
            subspot_distances = torch.tensor(subspot_distances, dtype=torch.float32).to(self.device)
        
        if neighbor_distances is not None:
            neighbor_distances = torch.tensor(neighbor_distances, dtype=torch.float32).to(self.device)
        
        if spatial_coordinates is not None:
            spatial_coordinates = torch.tensor(spatial_coordinates, dtype=torch.float32).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Run prediction
        with torch.no_grad():
            outputs = self.model(
                spot_features=spot_features,
                subspot_features=subspot_features,
                neighbor_features=neighbor_features,
                subspot_distances=subspot_distances,
                neighbor_distances=neighbor_distances,
                spatial_coordinates=spatial_coordinates
            )
        
        # Convert outputs to numpy arrays
        results = {
            'measured_predictions': outputs['measured_predictions'].cpu().numpy(),
            'unmeasured_predictions': outputs['unmeasured_predictions'].cpu().numpy(),
            'cell_type_probs': outputs['cell_type_probs'].cpu().numpy()
        }
        
        return results
    
    def rank_genes(self, gene_expression: np.ndarray, cell_labels: np.ndarray, 
                  gene_names: List[str], output_dir: str) -> Dict:
        """
        Rank genes based on their ability to differentiate between tissue types.
        
        Args:
            gene_expression: Gene expression matrix of shape [n_cells, n_genes]
            cell_labels: Cell labels of shape [n_cells], with values 'dysplastic' or 'non_dysplastic'
            gene_names: List of gene names
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing gene ranking results
        """
        # Calculate log fold change and perform statistical testing
        ranking_results = self.gene_ranking.calculate_logfc(gene_expression, cell_labels)
        
        # Create volcano plot
        import os
        os.makedirs(output_dir, exist_ok=True)
        volcano_plot_path = os.path.join(output_dir, 'volcano_plot.png')
        self.gene_ranking.create_volcano_plot(
            ranking_results['log_fold_change'],
            ranking_results['p_values'],
            gene_names,
            volcano_plot_path
        )
        
        # Save ranked gene list
        ranked_genes_path = os.path.join(output_dir, 'ranked_genes.csv')
        try:
            import pandas as pd
            ranked_indices = ranking_results['ranked_genes']
            ranked_df = pd.DataFrame({
                'gene': [gene_names[i] for i in ranked_indices],
                'log_fold_change': ranking_results['log_fold_change'][ranked_indices],
                'p_value': ranking_results['p_values'][ranked_indices],
                'q_value': ranking_results['q_values'][ranked_indices]
            })
            ranked_df.to_csv(ranked_genes_path, index=False)
        except:
            # Fallback if pandas is not available
            with open(ranked_genes_path, 'w') as f:
                f.write('gene,log_fold_change,p_value,q_value\n')
                for i in ranking_results['ranked_genes']:
                    f.write(f"{gene_names[i]},{ranking_results['log_fold_change'][i]},{ranking_results['p_values'][i]},{ranking_results['q_values'][i]}\n")
        
        return ranking_results
    
    def run_pipeline(self, data: Dict, output_dir: str) -> Dict:
        """
        Run the complete pipeline from raw data to gene ranking.
        
        Args:
            data: Dictionary containing:
                - spot_features: Spot-level features
                - subspot_features: Sub-spot features
                - neighbor_features: Neighbor features
                - subspot_distances: Distances to sub-spots
                - neighbor_distances: Distances to neighbors
                - spatial_coordinates: Spatial coordinates
                - cell_labels: Cell labels (dysplastic or non_dysplastic)
                - gene_names: List of gene names
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary containing pipeline results
        """
        # Step 1: Predict gene expression (DeepSpot + Tarandros)
        prediction_results = self.predict(
            spot_features=data['spot_features'],
            subspot_features=data['subspot_features'],
            neighbor_features=data['neighbor_features'],
            subspot_distances=data.get('subspot_distances'),
            neighbor_distances=data.get('neighbor_distances'),
            spatial_coordinates=data.get('spatial_coordinates')
        )
        
        # Step 2: Rank genes (LogFC)
        # Combine measured and unmeasured gene predictions
        all_predictions = np.concatenate([
            prediction_results['measured_predictions'],
            prediction_results['unmeasured_predictions']
        ], axis=1)
        
        # Combine gene names
        all_gene_names = data['gene_names']
        
        # Rank genes
        ranking_results = self.rank_genes(
            gene_expression=all_predictions,
            cell_labels=data['cell_labels'],
            gene_names=all_gene_names,
            output_dir=output_dir
        )
        
        # Combine results
        results = {
            'prediction_results': prediction_results,
            'ranking_results': ranking_results
        }
        
        return results

# Loss functions for training

def spearman_correlation_loss(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Differentiable approximation of Spearman correlation loss.
    
    Args:
        predictions: Predicted values of shape [batch_size, n_features]
        targets: Target values of shape [batch_size, n_features]
        eps: Small constant to avoid division by zero
        
    Returns:
        Loss value (1 - mean Spearman correlation)
    """
    # Convert to ranks along feature dimension (dim=1)
    def _to_ranks(x):
        ranks = torch.argsort(torch.argsort(x, dim=1), dim=1).float()
        return ranks
    
    pred_ranks = _to_ranks(predictions)
    target_ranks = _to_ranks(targets)
    
    # Calculate mean rank for each sample
    pred_mean = pred_ranks.mean(dim=1, keepdim=True)
    target_mean = target_ranks.mean(dim=1, keepdim=True)
    
    # Calculate differences from mean
    pred_diff = pred_ranks - pred_mean
    target_diff = target_ranks - target_mean
    
    # Calculate covariance
    cov = (pred_diff * target_diff).sum(dim=1)
    
    # Calculate standard deviations
    pred_std = torch.sqrt((pred_diff ** 2).sum(dim=1) + eps)
    target_std = torch.sqrt((target_diff ** 2).sum(dim=1) + eps)
    
    # Calculate correlation
    correlation = cov / (pred_std * target_std + eps)
    
    # Return loss (1 - mean correlation)
    return 1.0 - correlation.mean()

def cell_wise_spearman_loss(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cell-wise Spearman correlation loss.
    
    This loss calculates the Spearman correlation for each cell (row) and averages across cells.
    It prioritizes preserving the relative expression levels of different genes within each cell.
    
    Args:
        predictions: Predicted values of shape [batch_size, n_genes]
        targets: Target values of shape [batch_size, n_genes]
        eps: Small constant to avoid division by zero
        
    Returns:
        Loss value (1 - mean cell-wise Spearman correlation)
    """
    # Convert to ranks along gene dimension (dim=1)
    def _to_ranks(x):
        ranks = torch.argsort(torch.argsort(x, dim=1), dim=1).float()
        return ranks
    
    pred_ranks = _to_ranks(predictions)
    target_ranks = _to_ranks(targets)
    
    # Calculate mean rank for each cell
    pred_mean = pred_ranks.mean(dim=1, keepdim=True)
    target_mean = target_ranks.mean(dim=1, keepdim=True)
    
    # Calculate differences from mean
    pred_diff = pred_ranks - pred_mean
    target_diff = target_ranks - target_mean
    
    # Calculate covariance
    cov = (pred_diff * target_diff).sum(dim=1)
    
    # Calculate standard deviations
    pred_std = torch.sqrt((pred_diff ** 2).sum(dim=1) + eps)
    target_std = torch.sqrt((target_diff ** 2).sum(dim=1) + eps)
    
    # Calculate correlation
    correlation = cov / (pred_std * target_std + eps)
    
    # Return loss (1 - mean correlation)
    return 1.0 - correlation.mean()

def gene_wise_spearman_loss(predictions: torch.Tensor, targets: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Gene-wise Spearman correlation loss.
    
    This loss calculates the Spearman correlation for each gene (column) and averages across genes.
    It prioritizes preserving the relative expression levels of each gene across different cells.
    
    Args:
        predictions: Predicted values of shape [batch_size, n_genes]
        targets: Target values of shape [batch_size, n_genes]
        eps: Small constant to avoid division by zero
        
    Returns:
        Loss value (1 - mean gene-wise Spearman correlation)
    """
    # Transpose to get genes as rows
    predictions_t = predictions.t()  # [n_genes, batch_size]
    targets_t = targets.t()  # [n_genes, batch_size]
    
    # Convert to ranks along cell dimension (dim=1)
    def _to_ranks(x):
        ranks = torch.argsort(torch.argsort(x, dim=1), dim=1).float()
        return ranks
    
    pred_ranks = _to_ranks(predictions_t)
    target_ranks = _to_ranks(targets_t)
    
    # Calculate mean rank for each gene
    pred_mean = pred_ranks.mean(dim=1, keepdim=True)
    target_mean = target_ranks.mean(dim=1, keepdim=True)
    
    # Calculate differences from mean
    pred_diff = pred_ranks - pred_mean
    target_diff = target_ranks - target_mean
    
    # Calculate covariance
    cov = (pred_diff * target_diff).sum(dim=1)
    
    # Calculate standard deviations
    pred_std = torch.sqrt((pred_diff ** 2).sum(dim=1) + eps)
    target_std = torch.sqrt((target_diff ** 2).sum(dim=1) + eps)
    
    # Calculate correlation
    correlation = cov / (pred_std * target_std + eps)
    
    # Return loss (1 - mean correlation)
    return 1.0 - correlation.mean()

def balanced_spearman_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                          gene_wise_weight: float = 0.3, cell_wise_weight: float = 0.7,
                          eps: float = 1e-8) -> torch.Tensor:
    """
    Balanced Spearman correlation loss that combines gene-wise and cell-wise metrics.
    
    Args:
        predictions: Predicted values of shape [batch_size, n_genes]
        targets: Target values of shape [batch_size, n_genes]
        gene_wise_weight: Weight for gene-wise Spearman loss
        cell_wise_weight: Weight for cell-wise Spearman loss
        eps: Small constant to avoid division by zero
        
    Returns:
        Weighted combination of gene-wise and cell-wise Spearman losses
    """
    gene_loss = gene_wise_spearman_loss(predictions, targets, eps)
    cell_loss = cell_wise_spearman_loss(predictions, targets, eps)
    
    return gene_wise_weight * gene_loss + cell_wise_weight * cell_loss

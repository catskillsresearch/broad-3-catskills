#!/usr/bin/env python3
# models/deepspot_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SubspotPhiNetwork(nn.Module):
    """
    Phi network for processing sub-spots independently.
    
    This network processes each sub-spot independently and extracts features
    that capture the local tissue structure.
    """
    def __init__(self, input_dim, hidden_dim):
        super(SubspotPhiNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch_size, num_subspots, input_dim)
        batch_size, num_subspots, input_dim = x.shape
        
        # Reshape to process all subspots at once
        x = x.view(-1, input_dim)
        
        # Apply Phi network
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Reshape back to (batch_size, num_subspots, hidden_dim)
        x = x.view(batch_size, num_subspots, -1)
        
        return x


class NeighborPhiNetwork(nn.Module):
    """
    Phi network for processing neighboring cells independently.
    
    This network processes each neighboring cell independently and extracts features
    that capture the broader tissue context.
    """
    def __init__(self, input_dim, hidden_dim):
        super(NeighborPhiNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x shape: (batch_size, num_neighbors, input_dim)
        batch_size, num_neighbors, input_dim = x.shape
        
        # Reshape to process all neighbors at once
        x = x.view(-1, input_dim)
        
        # Apply Phi network
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Reshape back to (batch_size, num_neighbors, hidden_dim)
        x = x.view(batch_size, num_neighbors, -1)
        
        return x


class SubspotAggregation(nn.Module):
    """
    Aggregation module for sub-spots using attention mechanism.
    
    This module aggregates sub-spot features using a weighted sum,
    where the weights are learned by an attention mechanism.
    """
    def __init__(self, hidden_dim):
        super(SubspotAggregation, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch_size, num_subspots, hidden_dim)
        
        # Compute attention weights
        attention_weights = F.softmax(self.attention(x), dim=1)
        
        # Apply attention weights
        x = torch.sum(x * attention_weights, dim=1)
        
        return x


class NeighborAggregation(nn.Module):
    """
    Aggregation module for neighbors using distance-aware attention mechanism.
    
    This module aggregates neighbor features using a weighted sum,
    where the weights are learned by an attention mechanism that takes
    into account the distance between cells.
    """
    def __init__(self, hidden_dim):
        super(NeighborAggregation, self).__init__()
        self.attention = nn.Linear(hidden_dim + 1, 1)  # +1 for distance
        
    def forward(self, x, distances):
        # x shape: (batch_size, num_neighbors, hidden_dim)
        # distances shape: (batch_size, num_neighbors)
        
        # Expand distances to match x's dimensions
        distances = distances.unsqueeze(-1)
        
        # Concatenate features and distances
        x_with_dist = torch.cat([x, distances], dim=-1)
        
        # Compute attention weights
        attention_weights = F.softmax(self.attention(x_with_dist), dim=1)
        
        # Apply attention weights
        x = torch.sum(x * attention_weights, dim=1)
        
        return x


class RhoNetwork(nn.Module):
    """
    Rho network for processing aggregated representations.
    
    This network processes the aggregated representations from the Phi networks
    to produce the final output.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads=4, dropout=0.3):
        super(RhoNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Multi-head attention for feature integration
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout)
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        
        # Apply Rho network
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Reshape for multi-head attention
        x_reshaped = x.unsqueeze(0)  # (1, batch_size, hidden_dim)
        
        # Apply multi-head attention
        attn_output, _ = self.multihead_attn(x_reshaped, x_reshaped, x_reshaped)
        
        # Reshape back
        attn_output = attn_output.squeeze(0)  # (batch_size, hidden_dim)
        
        # Final output layer
        output = self.fc_out(attn_output)
        
        return output


class DeepSpotModel(nn.Module):
    """
    DeepSpot model with deep-set architecture for spatial transcriptomics.
    
    This model integrates information from three spatial levels:
    - Spot level: Features from the immediate area around each cell
    - Sub-spot level: Features from smaller regions within the spot
    - Neighborhood level: Features from neighboring cells
    
    It uses a permutation-invariant deep-set architecture to handle
    variable numbers of sub-spots and neighbors.
    """
    def __init__(self, config):
        super(DeepSpotModel, self).__init__()
        
        # Extract parameters from config
        feature_dim = config.get('feature_dim', 512)
        phi_dim = config.get('phi_dim', 256)
        embedding_dim = config.get('embedding_dim', 512)
        n_heads = config.get('n_heads', 4)
        dropout = config.get('dropout', 0.3)
        n_genes = config.get('n_genes', 1000)  # This should be set based on actual data
        
        # Phi networks
        self.subspot_phi = SubspotPhiNetwork(feature_dim, phi_dim)
        self.neighbor_phi = NeighborPhiNetwork(feature_dim, phi_dim)
        
        # Aggregation modules
        self.subspot_aggregation = SubspotAggregation(phi_dim)
        self.neighbor_aggregation = NeighborAggregation(phi_dim)
        
        # Spot-level feature processing
        self.spot_fc = nn.Linear(feature_dim, phi_dim)
        
        # Feature integration
        self.integration_fc = nn.Linear(phi_dim * 3, embedding_dim)
        
        # Rho network
        self.rho = RhoNetwork(embedding_dim, embedding_dim, n_genes, n_heads, dropout)
        
    def forward(self, spot_features, subspot_features, neighbor_features, neighbor_distances=None):
        """
        Forward pass of the DeepSpot model.
        
        Args:
            spot_features: Tensor of shape (batch_size, feature_dim)
            subspot_features: Tensor of shape (batch_size, num_subspots, feature_dim)
            neighbor_features: Tensor of shape (batch_size, num_neighbors, feature_dim)
            neighbor_distances: Tensor of shape (batch_size, num_neighbors)
                               If None, distances are assumed to be uniform
        
        Returns:
            Tensor of shape (batch_size, n_genes) containing predicted gene expression
        """
        # Process spot features
        spot_processed = F.relu(self.spot_fc(spot_features))
        
        # Process sub-spot features
        subspot_processed = self.subspot_phi(subspot_features)
        subspot_aggregated = self.subspot_aggregation(subspot_processed)
        
        # Process neighbor features
        neighbor_processed = self.neighbor_phi(neighbor_features)
        
        # Create uniform distances if not provided
        if neighbor_distances is None:
            batch_size, num_neighbors = neighbor_features.shape[:2]
            neighbor_distances = torch.ones(batch_size, num_neighbors, device=neighbor_features.device)
        
        neighbor_aggregated = self.neighbor_aggregation(neighbor_processed, neighbor_distances)
        
        # Integrate features from all levels
        integrated_features = torch.cat([
            spot_processed,
            subspot_aggregated,
            neighbor_aggregated
        ], dim=1)
        
        # Apply integration layer
        integrated_features = F.relu(self.integration_fc(integrated_features))
        
        # Apply Rho network to get final predictions
        gene_expression = self.rho(integrated_features)
        
        return gene_expression


# Utility function for Spearman correlation loss
def spearman_correlation_loss(y_pred, y_true, epsilon=1e-8):
    """
    Differentiable approximation of Spearman rank correlation loss.
    
    Args:
        y_pred: Predicted values
        y_true: Ground truth values
        epsilon: Small constant for numerical stability
    
    Returns:
        Loss value (1 - Spearman correlation)
    """
    # Convert tensors to ranks
    def rank_tensor(x):
        # Sort the tensor
        sorted_x, indices = torch.sort(x, dim=1)
        # Get the ranks
        ranks = torch.zeros_like(x)
        batch_size = x.shape[0]
        for i in range(batch_size):
            ranks[i, indices[i]] = torch.arange(x.shape[1], device=x.device, dtype=torch.float32)
        return ranks
    
    # Get ranks
    rank_pred = rank_tensor(y_pred)
    rank_true = rank_tensor(y_true)
    
    # Compute mean of ranks
    mean_pred = torch.mean(rank_pred, dim=1, keepdim=True)
    mean_true = torch.mean(rank_true, dim=1, keepdim=True)
    
    # Center the ranks
    centered_pred = rank_pred - mean_pred
    centered_true = rank_true - mean_true
    
    # Compute covariance
    cov = torch.sum(centered_pred * centered_true, dim=1)
    
    # Compute standard deviations
    std_pred = torch.sqrt(torch.sum(centered_pred ** 2, dim=1) + epsilon)
    std_true = torch.sqrt(torch.sum(centered_true ** 2, dim=1) + epsilon)
    
    # Compute correlation
    correlation = cov / (std_pred * std_true)
    
    # Return loss (1 - correlation)
    return 1.0 - torch.mean(correlation)

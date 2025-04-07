#!/usr/bin/env python3
# models/logfc_gene_ranking.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import os

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
    
    def save_rankings(self, output_path: str) -> None:
        """
        Save gene rankings to CSV file.
        
        Args:
            output_path: Path to save the rankings
        """
        if self.gene_rankings is None:
            raise ValueError("Gene rankings have not been computed yet. Call rank_genes() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save rankings
        rankings_df = self.gene_rankings[['gene_name', 'rank', 'logFC']]
        rankings_df.to_csv(output_path, index=False)
        
        print(f"Gene rankings saved to {output_path}")
    
    def get_top_genes(self, n: int = 100) -> List[str]:
        """
        Get the top N genes based on absolute logFC.
        
        Args:
            n: Number of top genes to return
            
        Returns:
            List of top gene names
        """
        if self.gene_rankings is None:
            raise ValueError("Gene rankings have not been computed yet. Call rank_genes() first.")
        
        return self.gene_rankings['gene_name'].head(n).tolist()
    
    def get_bottom_genes(self, n: int = 100) -> List[str]:
        """
        Get the bottom N genes based on absolute logFC.
        
        Args:
            n: Number of bottom genes to return
            
        Returns:
            List of bottom gene names
        """
        if self.gene_rankings is None:
            raise ValueError("Gene rankings have not been computed yet. Call rank_genes() first.")
        
        return self.gene_rankings['gene_name'].tail(n).tolist()
    
    def get_upregulated_genes(self, threshold: float = 1.0) -> List[str]:
        """
        Get upregulated genes (logFC > threshold).
        
        Args:
            threshold: logFC threshold
            
        Returns:
            List of upregulated gene names
        """
        if self.gene_rankings is None:
            raise ValueError("Gene rankings have not been computed yet. Call rank_genes() first.")
        
        upregulated = self.gene_rankings[self.gene_rankings['logFC'] > threshold]
        return upregulated['gene_name'].tolist()
    
    def get_downregulated_genes(self, threshold: float = -1.0) -> List[str]:
        """
        Get downregulated genes (logFC < threshold).
        
        Args:
            threshold: logFC threshold
            
        Returns:
            List of downregulated gene names
        """
        if self.gene_rankings is None:
            raise ValueError("Gene rankings have not been computed yet. Call rank_genes() first.")
        
        downregulated = self.gene_rankings[self.gene_rankings['logFC'] < threshold]
        return downregulated['gene_name'].tolist()
    
    def visualize_rankings(self, output_path: str, top_n: int = 20) -> None:
        """
        Create visualization of gene rankings.
        
        Args:
            output_path: Path to save the visualization
            top_n: Number of top genes to visualize
        """
        if self.gene_rankings is None:
            raise ValueError("Gene rankings have not been computed yet. Call rank_genes() first.")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Get top genes
            top_genes = self.gene_rankings.head(top_n)
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Create bar plot
            sns.barplot(x='logFC', y='gene_name', data=top_genes.sort_values('logFC'))
            
            # Add title and labels
            plt.title(f'Top {top_n} Genes by Absolute Log Fold Change')
            plt.xlabel('Log Fold Change')
            plt.ylabel('Gene')
            
            # Add grid
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Tight layout
            plt.tight_layout()
            
            # Save figure
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Gene ranking visualization saved to {output_path}")
            
        except ImportError:
            print("Matplotlib and/or seaborn not available. Skipping visualization.")


class LogFCBasedFeatureSelection(nn.Module):
    """
    Feature selection module based on logFC rankings.
    This can be used to select the most informative genes for downstream tasks.
    """
    def __init__(self, n_genes: int, n_selected: int):
        """
        Initialize the feature selection module.
        
        Args:
            n_genes: Total number of genes
            n_selected: Number of genes to select
        """
        super().__init__()
        self.n_genes = n_genes
        self.n_selected = n_selected
        self.register_buffer('selection_mask', torch.zeros(n_genes))
        
    def set_selected_indices(self, indices: List[int]) -> None:
        """
        Set the indices of selected genes.
        
        Args:
            indices: List of gene indices to select
        """
        mask = torch.zeros(self.n_genes, device=self.selection_mask.device)
        mask[indices] = 1.0
        self.selection_mask = mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feature selection to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, n_genes, ...)
            
        Returns:
            Tensor with only selected genes
        """
        # Apply selection mask
        selected_indices = torch.nonzero(self.selection_mask).squeeze(-1)
        return torch.index_select(x, 1, selected_indices)


def compute_gene_importance_scores(model, data_loader, device='cuda'):
    """
    Compute gene importance scores based on model gradients.
    This can be used as an alternative or complement to logFC ranking.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device to use
        
    Returns:
        Array of importance scores for each gene
    """
    model.eval()
    importance_scores = []
    
    for batch in data_loader:
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Forward pass with gradient computation
        model.zero_grad()
        outputs = model(**batch)
        loss = outputs['loss']
        loss.backward()
        
        # Get gradients for gene embeddings
        if hasattr(model, 'gene_embeddings'):
            grad = model.gene_embeddings.weight.grad.abs().mean(dim=1).cpu().numpy()
            importance_scores.append(grad)
    
    # Average importance scores across batches
    if importance_scores:
        return np.mean(np.stack(importance_scores), axis=0)
    else:
        return None

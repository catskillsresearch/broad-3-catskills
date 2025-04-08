import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import wandb

class VisualizationModule:
    """
    A comprehensive visualization module for the Luigi pipeline.
    This module integrates volcano plots and visualizations from visualize_synthetic_dataset.py
    and ensures all visualizations are logged to wandb.
    """
    
    def __init__(self, output_dir, experiment_name=None, use_wandb=True):
        """
        Initialize the visualization module.
        
        Args:
            output_dir: Directory to save visualizations
            experiment_name: Name of the experiment for wandb logging
            use_wandb: Whether to log visualizations to wandb
        """
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        
        # Create visualization directory
        self.vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Initialize wandb if needed
        if self.use_wandb and not wandb.run:
            wandb.init(
                project="spatial-transcriptomics",
                name=self.experiment_name,
                dir=output_dir
            )
    
    def create_volcano_plot(self, expression_data, region_labels, gene_names=None, 
                           p_threshold=0.05, fc_threshold=1.0, figsize=(10, 8)):
        """
        Create volcano plot for differential expression analysis.
        
        Args:
            expression_data: Gene expression data (n_cells, n_genes)
            region_labels: Region labels for cells (n_cells)
            gene_names: List of gene names
            p_threshold: P-value threshold for significance
            fc_threshold: Fold change threshold for significance
            figsize: Figure size
            
        Returns:
            Path to saved visualization
        """
        print("Creating volcano plot...")
        
        # Convert region labels to binary (dysplastic vs non-dysplastic)
        if np.unique(region_labels).size > 2:
            print("Converting region labels to binary (dysplastic vs non-dysplastic)...")
            # Assume region 0 is dysplastic, all others are non-dysplastic
            binary_labels = (region_labels == 0).astype(int)
        else:
            binary_labels = region_labels
        
        # Split cells by region
        dysplastic_mask = binary_labels == 1
        non_dysplastic_mask = binary_labels == 0
        
        dysplastic_expr = expression_data[dysplastic_mask]
        non_dysplastic_expr = expression_data[non_dysplastic_mask]
        
        print(f"Number of dysplastic cells: {dysplastic_expr.shape[0]}")
        print(f"Number of non-dysplastic cells: {non_dysplastic_expr.shape[0]}")
        
        # Initialize results
        results = []
        
        # Calculate statistics for each gene
        for i in range(expression_data.shape[1]):
            # Get expression values for this gene
            dysplastic_gene = dysplastic_expr[:, i]
            non_dysplastic_gene = non_dysplastic_expr[:, i]
            
            # Calculate mean expression in each region
            mean_dysplastic = np.mean(dysplastic_gene)
            mean_non_dysplastic = np.mean(non_dysplastic_gene)
            
            # Calculate log fold change
            epsilon = 1e-10  # Small constant to avoid log(0)
            logFC = np.log2((mean_dysplastic + epsilon) / (mean_non_dysplastic + epsilon))
            
            # Calculate p-value using t-test
            t_stat, p_value = stats.ttest_ind(dysplastic_gene, non_dysplastic_gene)
            
            # Store results
            gene_name = gene_names[i] if gene_names is not None else f"Gene_{i}"
            results.append({
                'Gene': gene_name,
                'logFC': logFC,
                'p_value': p_value,
                'mean_dysplastic': mean_dysplastic,
                'mean_non_dysplastic': mean_non_dysplastic
            })
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Add -log10(p-value)
        results_df['-log10(p)'] = -np.log10(results_df['p_value'])
        
        # Create figure
        fig, ax = plt.figure(figsize=figsize)
        
        # Plot all points
        ax.scatter(
            results_df['logFC'],
            results_df['-log10(p)'],
            alpha=0.7,
            s=10,
            color='gray'
        )
        
        # Highlight significant points
        significant = (results_df['p_value'] < p_threshold) & (np.abs(results_df['logFC']) > fc_threshold)
        
        ax.scatter(
            results_df.loc[significant, 'logFC'],
            results_df.loc[significant, '-log10(p)'],
            alpha=0.7,
            s=20,
            color='red'
        )
        
        # Add threshold lines
        ax.axhline(-np.log10(p_threshold), color='blue', linestyle='--')
        ax.axvline(fc_threshold, color='blue', linestyle='--')
        ax.axvline(-fc_threshold, color='blue', linestyle='--')
        
        # Label top genes
        top_genes = results_df.loc[significant].sort_values('-log10(p)', ascending=False).head(10)
        
        for _, row in top_genes.iterrows():
            ax.text(
                row['logFC'],
                row['-log10(p)'],
                row['Gene'],
                fontsize=8,
                ha='center',
                va='bottom'
            )
        
        ax.set_title('Volcano Plot: Dysplastic vs. Non-dysplastic Regions')
        ax.set_xlabel('log2(Fold Change)')
        ax.set_ylabel('-log10(p-value)')
        
        # Add text annotations for quadrants
        ax.text(
            np.max(results_df['logFC']) * 0.8,
            np.max(results_df['-log10(p)']) * 0.8,
            'Up in Dysplastic',
            fontsize=10,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        ax.text(
            np.min(results_df['logFC']) * 0.8,
            np.max(results_df['-log10(p)']) * 0.8,
            'Down in Dysplastic',
            fontsize=10,
            ha='center',
            va='center',
            bbox=dict(facecolor='white', alpha=0.7)
        )
        
        # Save figure
        save_path = os.path.join(self.vis_dir, 'volcano_plot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({"volcano_plot": wandb.Image(save_path)})
        
        return save_path
    
    def create_dataset_overview(self, gene_expression, cell_coordinates, region_labels, 
                               quality_scores=None, gene_names=None):
        """
        Create dataset overview visualizations.
        
        Args:
            gene_expression: Gene expression data (n_cells, n_genes)
            cell_coordinates: Cell coordinates (n_cells, 2)
            region_labels: Region labels for cells (n_cells)
            quality_scores: Quality scores for cells (n_cells)
            gene_names: List of gene names
            
        Returns:
            Path to saved visualization
        """
        print("Creating dataset overview visualizations...")
        
        # Create a figure with multiple subplots for overview
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Spatial distribution of cells colored by region
        ax1 = fig.add_subplot(2, 3, 1)
        scatter = ax1.scatter(cell_coordinates[:, 0], cell_coordinates[:, 1], 
                             c=region_labels, cmap='tab10', s=10, alpha=0.7)
        ax1.set_title('Spatial Distribution by Region')
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        plt.colorbar(scatter, ax=ax1, label='Region')
        
        # 2. Spatial distribution of cells colored by quality score (if available)
        ax2 = fig.add_subplot(2, 3, 2)
        if quality_scores is not None:
            scatter = ax2.scatter(cell_coordinates[:, 0], cell_coordinates[:, 1], 
                                 c=quality_scores, cmap='viridis', s=10, alpha=0.7)
            ax2.set_title('Spatial Distribution by Quality Score')
            plt.colorbar(scatter, ax=ax2, label='Quality Score')
        else:
            # Use first gene expression as example
            scatter = ax2.scatter(cell_coordinates[:, 0], cell_coordinates[:, 1], 
                                 c=gene_expression[:, 0], cmap='viridis', s=10, alpha=0.7)
            gene_label = gene_names[0] if gene_names is not None else "Gene 0"
            ax2.set_title(f'Spatial Distribution of {gene_label}')
            plt.colorbar(scatter, ax=ax2, label='Expression')
        
        ax2.set_xlabel('X Coordinate')
        ax2.set_ylabel('Y Coordinate')
        
        # 3. PCA of gene expression colored by region
        ax3 = fig.add_subplot(2, 3, 3)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(gene_expression)
        scatter = ax3.scatter(pca_result[:, 0], pca_result[:, 1], 
                             c=region_labels, cmap='tab10', s=10, alpha=0.7)
        ax3.set_title('PCA of Gene Expression by Region')
        ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.colorbar(scatter, ax=ax3, label='Region')
        
        # 4. Gene expression distribution
        ax4 = fig.add_subplot(2, 3, 4)
        # Sample a few genes for visualization
        sample_genes = np.random.choice(gene_expression.shape[1], min(5, gene_expression.shape[1]), replace=False)
        for gene_idx in sample_genes:
            gene_label = gene_names[gene_idx] if gene_names is not None else f"Gene {gene_idx}"
            sns.kdeplot(gene_expression[:, gene_idx], ax=ax4, label=gene_label)
        ax4.set_title('Gene Expression Distributions')
        ax4.set_xlabel('Expression Level')
        ax4.set_ylabel('Density')
        ax4.legend()
        
        # 5. Gene-gene correlation heatmap
        ax5 = fig.add_subplot(2, 3, 5)
        # Sample a subset of genes for correlation heatmap
        n_sample_genes = min(20, gene_expression.shape[1])
        sample_genes_idx = np.random.choice(gene_expression.shape[1], n_sample_genes, replace=False)
        corr_matrix = np.corrcoef(gene_expression[:, sample_genes_idx].T)
        im = ax5.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax5.set_title('Gene-Gene Correlation Matrix')
        ax5.set_xticks([])
        ax5.set_yticks([])
        plt.colorbar(im, ax=ax5, label='Correlation')
        
        # 6. Dataset statistics
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.axis('off')
        stats_text = (
            f"Dataset Statistics:\n\n"
            f"Number of Cells: {gene_expression.shape[0]}\n"
            f"Number of Genes: {gene_expression.shape[1]}\n"
            f"Number of Regions: {np.unique(region_labels).size}\n"
        )
        ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
                 fontsize=12, verticalalignment='top')
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, 'dataset_overview.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({"dataset_overview": wandb.Image(save_path)})
        
        return save_path
    
    def create_spatial_expression_map(self, cell_coordinates, expression_values, gene_names=None, 
                                     n_genes=4, figsize=(15, 12)):
        """
        Create spatial expression map visualizations.
        
        Args:
            cell_coordinates: Cell coordinates (n_cells, 2)
            expression_values: Gene expression data (n_cells, n_genes)
            gene_names: List of gene names
            n_genes: Number of genes to visualize
            figsize: Figure size
            
        Returns:
            Path to saved visualization
        """
        print("Creating spatial expression map visualizations...")
        
        if gene_names is None:
            # Select genes with highest spatial variance
            spatial_var = []
            for i in range(expression_values.shape[1]):
                spatial_var.append(np.var(expression_values[:, i]))
            
            top_genes = np.argsort(spatial_var)[-n_genes:]
            gene_labels = [f'Gene {i}' for i in top_genes]
        else:
            if len(gene_names) > n_genes:
                # Select first n_genes
                gene_names = gene_names[:n_genes]
            top_genes = range(len(gene_names))
            gene_labels = gene_names
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, (gene_idx, gene_label) in enumerate(zip(top_genes, gene_labels)):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Get expression values for this gene
            expr = expression_values[:, gene_idx]
            
            # Plot cells colored by expression
            scatter = ax.scatter(
                cell_coordinates[:, 0],
                cell_coordinates[:, 1],
                c=expr,
                cmap='viridis',
                s=10,
                alpha=0.7
            )
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, label='Expression')
            
            ax.set_title(f'{gene_label} Expression')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_aspect('equal')
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, 'spatial_expression_map.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({"spatial_expression_map": wandb.Image(save_path)})
        
        return save_path
    
    def create_prediction_accuracy_visualization(self, predictions, targets, gene_names=None, 
                                               n_genes=5, figsize=(15, 12)):
        """
        Create prediction accuracy visualizations.
        
        Args:
            predictions: Predicted gene expression (n_cells, n_genes)
            targets: Target gene expression (n_cells, n_genes)
            gene_names: List of gene names
            n_genes: Number of genes to visualize
            figsize: Figure size
            
        Returns:
            Path to saved visualization
        """
        print("Creating prediction accuracy visualizations...")
        
        if gene_names is None:
            # Select random genes if not specified
            gene_names = np.random.choice(range(predictions.shape[1]), 
                                         size=min(n_genes, predictions.shape[1]), 
                                         replace=False)
            gene_labels = [f'Gene {i}' for i in gene_names]
        else:
            gene_labels = gene_names
            gene_names = [list(gene_names).index(g) for g in gene_names[:n_genes]]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Scatter plots of predicted vs. actual values
        ax = axes[0, 0]
        for i, gene_idx in enumerate(gene_names):
            ax.scatter(
                targets[:, gene_idx],
                predictions[:, gene_idx],
                alpha=0.5,
                label=gene_labels[i]
            )
        
        # Add diagonal line
        min_val = min(targets.min(), predictions.min())
        max_val = max(targets.max(), predictions.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--')
        
        ax.set_title('Predicted vs. Actual Expression')
        ax.set_xlabel('Actual Expression')
        ax.set_ylabel('Predicted Expression')
        ax.legend()
        
        # Histogram of prediction errors
        ax = axes[0, 1]
        for i, gene_idx in enumerate(gene_names):
            errors = predictions[:, gene_idx] - targets[:, gene_idx]
            sns.histplot(errors, kde=True, ax=ax, label=gene_labels[i], alpha=0.5)
        
        ax.set_title('Prediction Error Distribution')
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Box plot of Spearman correlations across genes
        ax = axes[1, 0]
        
        # Compute cell-wise Spearman correlations
        cell_correlations = []
        for i in range(predictions.shape[0]):
            corr, _ = stats.spearmanr(predictions[i, :], targets[i, :])
            cell_correlations.append(corr)
        
        # Compute gene-wise Spearman correlations
        gene_correlations = []
        for i in range(predictions.shape[1]):
            corr, _ = stats.spearmanr(predictions[:, i], targets[:, i])
            gene_correlations.append(corr)
        
        # Create box plots
        data = [cell_correlations, gene_correlations]
        labels = ['Cell-wise', 'Gene-wise']
        
        sns.boxplot(data=data, ax=ax)
        ax.set_title('Spearman Correlation Distribution')
        ax.set_xticklabels(labels)
        ax.set_ylabel('Spearman Correlation')
        
        # Heatmap of correlation matrix for selected genes
        ax = axes[1, 1]
        
        # Compute correlation matrix
        corr_matrix = np.zeros((len(gene_names), len(gene_names)))
        for i, gene_i in enumerate(gene_names):
            for j, gene_j in enumerate(gene_names):
                corr, _ = stats.spearmanr(predictions[:, gene_i], predictions[:, gene_j])
                corr_matrix[i, j] = corr
        
        # Plot heatmap
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax,
                    xticklabels=gene_labels, yticklabels=gene_labels)
        ax.set_title('Correlation Matrix of Predicted Expression')
        
        plt.tight_layout()
        save_path = os.path.join(self.vis_dir, 'prediction_accuracy.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to wandb
        if self.use_wandb:
            wandb.log({"prediction_accuracy": wandb.Image(save_path)})
        
        return save_path
    
    def create_dimensionality_reduction_visualizations(self, predictions, targets, vis_dir):
        """
        Create dimensionality reduction visualizations (PCA, t-SNE, UMAP).
        
        Args:
            predictions: Predicted gene expression (n_cells, n_genes)
            targets: Target gene expression (n_cells, n_genes)
            vis_dir: Directory to save visualizations
            
        Returns:
            Dictionary of paths to saved visualizations
        """
        print("Creating dimensionality reduction visualizations...")
        
        vis_paths = {}
        
        # 1. PCA Visualization
        try:
            pca_vis_path = self._create_pca_visualization(predictions, targets, vis_dir)
            vis_paths['pca_visualization'] = pca_vis_path
        except Exception as e:
            print(f"Warning: Could not create PCA visualization: {e}")
            vis_paths['pca_visualization'] = None
        
        # 2. t-SNE Visualization
        try:
            tsne_vis_path = self._create_tsne_visualization(predictions, targets, vis_dir)
            vis_paths['tsne_visualization'] = tsne_vis_path
        except Exception as e:
            print(f"Warning: Could not create t-SNE visualization: {e}")
            vis_paths['tsne_visualization'] = None
        
        # 3. UMAP Visualization
        try:
            umap_vis_path = self._create_umap_visualization(predictions, targets, vis_dir)
            vis_paths['umap_visualization'] = umap_vis_path
        except Exception as e:
            print(f"Warning: Could not create UMAP visualization: {e}")
            vis_paths['umap_visualization'] = None
        
        # Log to wandb
        if self.use_wandb:
            log_dict = {}
            for key, path in vis_paths.items():
                if path is not None:
                    log_dict[key] = wandb.Image(path)
            
            if log_dict:
                wandb.log(log_dict)
        
        return vis_paths
    
    def _create_pca_visualization(self, predictions, targets, vis_dir):
        """Create PCA visualization of predictions vs targets."""
        # Apply PCA
        pca = PCA(n_components=2)
        pca_pred = pca.fit_transform(predictions)
        pca_target = pca.transform(targets)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot predictions
        plt.scatter(pca_pred[:, 0], pca_pred[:, 1], c='blue', alpha=0.5, label='Predictions')
        
        # Plot targets
        plt.scatter(pca_target[:, 0], pca_target[:, 1], c='red', alpha=0.5, label='Ground Truth')
        
        # Add arrows connecting corresponding points
        for i in range(len(pca_pred)):
            plt.arrow(pca_target[i, 0], pca_target[i, 1], 
                     pca_pred[i, 0] - pca_target[i, 0], 
                     pca_pred[i, 1] - pca_target[i, 1], 
                     color='gray', alpha=0.3, width=0.001, head_width=0.01)
        
        plt.title('PCA: Predictions vs Ground Truth')
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        save_path = os.path.join(vis_dir, 'pca_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _create_tsne_visualization(self, predictions, targets, vis_dir):
        """Create t-SNE visualization of predictions vs targets."""
        # Check if we have enough samples for t-SNE
        n_samples = len(predictions) + len(targets)
        
        # Calculate appropriate perplexity (should be less than n_samples)
        # Recommended range is 5-50, but must be less than n_samples
        perplexity = min(30, max(5, n_samples // 5))
        
        # If we still don't have enough samples, raise an exception
        if perplexity >= n_samples:
            raise ValueError(f"Not enough samples for t-SNE visualization. Need at least 6 samples, got {n_samples}.")
        
        # Apply t-SNE with adjusted perplexity
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        combined_data = np.vstack([predictions, targets])
        tsne_result = tsne.fit_transform(combined_data)
        
        # Split results back into predictions and targets
        tsne_pred = tsne_result[:len(predictions)]
        tsne_target = tsne_result[len(predictions):]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot predictions
        plt.scatter(tsne_pred[:, 0], tsne_pred[:, 1], c='blue', alpha=0.5, label='Predictions')
        
        # Plot targets
        plt.scatter(tsne_target[:, 0], tsne_target[:, 1], c='red', alpha=0.5, label='Ground Truth')
        
        # Add arrows connecting corresponding points
        for i in range(len(tsne_pred)):
            plt.arrow(tsne_target[i, 0], tsne_target[i, 1], 
                     tsne_pred[i, 0] - tsne_target[i, 0], 
                     tsne_pred[i, 1] - tsne_target[i, 1], 
                     color='gray', alpha=0.3, width=0.001, head_width=0.01)
        
        plt.title(f't-SNE (perplexity={perplexity}): Predictions vs Ground Truth')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        save_path = os.path.join(vis_dir, 'tsne_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def _create_umap_visualization(self, predictions, targets, vis_dir):
        """Create UMAP visualization of predictions vs targets."""
        # Check if we have enough samples for UMAP
        n_samples = len(predictions) + len(targets)
        
        # UMAP requires at least 2 samples
        if n_samples < 2:
            raise ValueError(f"Not enough samples for UMAP visualization. Need at least 2 samples, got {n_samples}.")
        
        # Adjust n_neighbors parameter based on dataset size
        n_neighbors = min(15, max(2, n_samples // 5))
        
        # Apply UMAP with adjusted parameters
        reducer = umap.UMAP(random_state=42, n_neighbors=n_neighbors)
        combined_data = np.vstack([predictions, targets])
        umap_result = reducer.fit_transform(combined_data)
        
        # Split results back into predictions and targets
        umap_pred = umap_result[:len(predictions)]
        umap_target = umap_result[len(predictions):]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot predictions
        plt.scatter(umap_pred[:, 0], umap_pred[:, 1], c='blue', alpha=0.5, label='Predictions')
        
        # Plot targets
        plt.scatter(umap_target[:, 0], umap_target[:, 1], c='red', alpha=0.5, label='Ground Truth')
        
        # Add arrows connecting corresponding points
        for i in range(len(umap_pred)):
            plt.arrow(umap_target[i, 0], umap_target[i, 1], 
                     umap_pred[i, 0] - umap_target[i, 0], 
                     umap_pred[i, 1] - umap_target[i, 1], 
                     color='gray', alpha=0.3, width=0.001, head_width=0.01)
        
        plt.title(f'UMAP (n_neighbors={n_neighbors}): Predictions vs Ground Truth')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save figure
        save_path = os.path.join(vis_dir, 'umap_visualization.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

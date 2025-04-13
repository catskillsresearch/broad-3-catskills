"""

The `gene_ranking` function identifies the top discriminating genes that differentiate dysplastic (pre-cancerous) from non-cancerous mucosa regions.

**Steps:**
1. **Input Predictions**: Take gene expression data for non-cancerous and dysplastic cells (shapes: n_cells x 18615genes v.s. m_cells x 18615genes).
2. **Compute Metrics**: Calculate mean, variance and absolute log fold change (logFC) for all 18 615 protein-coding genes.
3. **Rank Genes**: Rank genes by their ability to distinguish dysplastic from non-cancerous regions using absolute log fold change.
4. **Output**: Return a DataFrame of gene names ranked from best to worst discriminator.

"""

import numpy as np
import pandas as pd

def gene_ranking(prediction_cell_ids_no_cancer, prediction_cell_ids_cancer,
                 column_for_ranking="abs_logFC", ascending=False):
    """
    Rank all 18,615 protein-coding genes based on their ability to distinguish dysplasia
    from non-cancerous mucosa regions. Each gene is assigned a rank from 1
    (best discriminator) to 18,615 (worst), comparing expression levels between
    non-cancerous and dysplastic tissue cells.

    Parameters:
    prediction_cell_ids_no_cancer: np.ndarray
        Predicted gene expression for cells from non-cancerous regions.
    prediction_cell_ids_cancer: np.ndarray
        Predicted gene expression for cells from dysplastic (pre-cancerous) regions.
    column_for_ranking: str
        Column name used to rank genes (default is "abs_logFC").
    ascending: bool
        Whether to sort in ascending order (default is False).

    Returns:
    pd.DataFrame
        A DataFrame containing gene names ranked by the specified metric.
    """

    # Calculate mean and variance for each gene
    mean_no_cancer = prediction_cell_ids_no_cancer.mean(axis=0)
    mean_cancer = prediction_cell_ids_cancer.mean(axis=0)

    var_no_cancer = prediction_cell_ids_no_cancer.var(axis=0)
    var_cancer = prediction_cell_ids_cancer.var(axis=0)

    ### Compute ranking metrics ###
    # Compute the absolute difference in mean expression levels
    dif_abs_mean = np.abs(mean_no_cancer - mean_cancer)

    # Compute the log fold change (logFC) for each gene
    epsilon = 1e-6  # Small value to avoid division by zero
    log_fc = np.log2((mean_cancer + epsilon) / (mean_no_cancer + epsilon))

    gene_ranking_df = pd.DataFrame({
        'mean_no_cancer': mean_no_cancer,
        'mean_cancer': mean_cancer,
        'variance_no_cancer': var_no_cancer,
        'variance_cancer': var_cancer,
        'dif_abs_mean': dif_abs_mean,
        'logFC': log_fc,
        'abs_logFC': np.abs(log_fc)
    })

    # Sort by column_for_ranking
    gene_ranking_df = gene_ranking_df.sort_values(by=column_for_ranking, ascending=ascending)

    print(f"Gene ranking by {column_for_ranking}:")
    print(gene_ranking_df.head())

    # Create the final ranked DataFrame with gene names and their ranks
    prediction = pd.DataFrame(
        gene_ranking_df.index,
        index=np.arange(1, len(gene_ranking_df) + 1),
        columns=['Gene Name'],
    )

    return prediction, gene_ranking_df
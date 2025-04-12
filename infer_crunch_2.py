from log1p_normalization_scale_factor import log1p_normalization_scale_factor
from find_matches_cos_similarity import find_matches_cos_similarity
import pandas as pd
import numpy as np

# 1. Prepare the necessary directories and load the configuration parameters from the previously trained model.
# 2. The test data, provided as a Zarr file, is read and specific subsets of the data (test and validation groups) are extracted.
# 3. Preprocess the data into image patches (X_test).
# 4. Generate embeddings for the test data and applies the trained models for regression predictions on the 460 measured genes.
# 5. Apply cosine similarities between predictions and single-cell RNA sequencing from shared genes.
# 6. Compute weighted average of unmeasured genes from scRNA-Seq based on similarity scores.
# 7. Format the predictions of the 2000 unmeasured genes for submission.

def infer_crunch_2(
    prediction_460_genes,  # Predicted expression values for 460 genes (Crunch 1 output)
    name_data: str,  # The name of the dataset being processed (only used for logging and directory naming)
    data_file_path: str,
    model_directory_path: str,  # Path to save the trained model and results
    scRNAseq,  # Single-cell RNA sequencing data (AnnData)
    filter_column: str = 'dysplasia',  # Column name used to filter scRNA-seq data
    filter_value: str = 'y'  # Value within the filter column used for filtering
):
    """
    Perform Crunch 2 inference: Predict the expression of 18,157 unmeasured genes in spatial
    transcriptomics data using the expression of the 460 inferred genes and single-cell RNA sequencing data for similarity-based matching.
    """

    ##############################################################################
    ##### BROAD Crunch 2: predict the expression of 18157 unmeasured genes #####
    print(f"\n** {name_data} Predict the expression of 18157 unmeasured genes (Crunch 2) ****************************************************************\n")

    print(f"--COSINE SIMILARITIES between {name_data} and single-cell RNA sequencing from shared genes--\n")

    # Get the 18615 genes to rank
    gene_18615_list = list(scRNAseq.var.index)

    # Extract shared and unmeasured genes
    gene_460 = list(prediction_460_genes.columns)
    common_genes = [g for g in gene_460 if g in gene_18615_list]
    print("Number of shared genes between scRNA-seq and xenium data:", len(common_genes))
    unmeasured_genes = [gene for gene in gene_18615_list if gene not in common_genes]
    print("Number of unmeasured genes in Xenium data:", len(unmeasured_genes), "\n")

    # scRNA-Seq data log1p-normalized with scale factor 10000 on 18615 genes
    rna_data_norm_10000_unmeasured_genes = scRNAseq[:, unmeasured_genes].X.toarray()
    # scRNA-Seq data log1p-normalized with scale factor 100 on 460 genes
    rna_data_norm_100_common_genes = log1p_normalization_scale_factor(scRNAseq[:, common_genes].layers["counts"].toarray(), scale_factor=100)
    print(f"scRNA-Seq data shape ({len(rna_data_norm_100_common_genes)} samples x {scRNAseq.X.shape[1]} shared genes + unmeasured genes)")
    print(f"Xenium data shape ({len(prediction_460_genes)} samples x {len(common_genes)} shared genes)")
    del scRNAseq

    # Similarity-Based Matching: Find the top_k most similar spots for each query
    top_k = 30
    print(f"\nCompute COSINE SIMILARITY: Find the top_k(={top_k}) similar scRNA-Seq cells for each Xenium cell...\n")
    matches, similarities = find_matches_cos_similarity(rna_data_norm_100_common_genes, prediction_460_genes[common_genes].values, top_k=top_k)
    del rna_data_norm_100_common_genes

    # Weighted Averaging of scRNA-Seq data log1p-normalized with scale factor 10000
    print("Compute WEIGHTED AVERAGE of unmeasured genes from scRNA-Seq based on similarity scores...")
    weighted_avg_df_10000 = pd.DataFrame([
        {
            **dict(zip(unmeasured_genes, np.average(rna_data_norm_10000_unmeasured_genes[indices, :], axis=0, weights=similarity).round(2)))  # updated
        }
        for i, (indices, similarity) in enumerate(zip(matches, similarities))
    ])
    weighted_avg_df_10000.index = prediction_460_genes.index

    # Free memory by deleting large variables and performing garbage collection
    del rna_data_norm_10000_unmeasured_genes, matches, similarities

    prediction_18615_genes = pd.concat([prediction_460_genes, weighted_avg_df_10000], axis=1)[gene_18615_list]

    print(f"\n-- {name_data} PREDICTION DONE --\n")

    # Return the final prediction DataFrame
    return prediction_18615_genes
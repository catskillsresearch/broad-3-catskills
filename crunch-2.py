"""
Infer function

The `infer` function loads trained models and performs inference on a new dataset.

1. Prepare the necessary directories and load the configuration parameters from the previously trained model.
2. The test data, provided as a Zarr file, is read and specific subsets of the data (test and validation groups) are extracted.
3. Preprocess the data into image patches (X_test).
4. Generate embeddings for the test data and applies the trained models for regression predictions on the 460 measured genes.
5. Apply cosine similarities between predictions and single-cell RNA sequencing from shared genes.
6. Compute weighted average of unmeasured genes from scRNA-Seq based on similarity scores.
7. Format the predictions of the 2000 unmeasured genes for submission.
"""

import os, json, torch, gc
from types import SimpleNamespace
import spatialdata as sd  # Manage multi-modal spatial omics datasets
from extract_spatial_positions import *
from infer1 import *
import torch.nn.functional as F
import numpy as np
import pandas as pd

from find_matches_cos_similarity import find_matches_cos_similarity

from log1p_normalization_scale_factor import log1p_normalization_scale_factor

# In the inference function, the trained model is loaded and used to make inferences on a
# sample of data that matches the characteristics of the training test.
def infer2(
    data_file_path: str,  # Path to a test dataset (in Zarr format) to perform inference on.
    model_directory_path: str  # Path to save the trained model and results
):
    ### Prepare Directories ###

    # Extract the name of the dataset from the file path (without extension)
    name_data = os.path.splitext(os.path.basename(data_file_path))[0]
    data_path = os.path.dirname(data_file_path)
    print(f"\n-- {name_data} INFERENCE ---------------------------------------------------------------\n")
    print(data_file_path)

    # Previous Directory to store models and results
    args, dir_processed_dataset, dir_models_and_results, list_ST_name_data = args1(model_directory_path)

    # Directory for processed test dataset (temporary storage)
    dir_processed_dataset_test = os.path.join("results", f"processed_dataset")
    os.makedirs(dir_processed_dataset_test, exist_ok=True)

    # Directory to store the test data embeddings (temporary storage)
    test_embed_dir = os.path.join(dir_processed_dataset_test, "ST_data_emb")
    os.makedirs(test_embed_dir, exist_ok=True)

    # Set device to GPU if available, else use CPU!!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##################################################################
    ##### BROAD Crunch 1: predict the expression of 460 genes #####
    print(f"\n** {name_data} Predict the expression of 460 genes (Crunch 1) ****************************************************************\n")

    ### Preprocess and Embedding Data + Regression inference ###

    # Read the spatial data from the provided file
    sdata = sd.read_zarr(data_file_path)

    # Extract cell IDs for test and validation groups
    cell_ids = list(sdata["cell_id-group"].obs.query("group == 'test' or group == 'validation'")["cell_id"])

    # Extract gene names from the spatial data
    gene_names = list(sdata["anucleus"].var.index)

    # Preprocess the test data for embedding (patch extraction)
    preprocess_spatial_transcriptomics_data_test(name_data, sdata, cell_ids, dir_processed_dataset_test,
                                                 args.target_patch_size, args.vis_width, args.show_extracted_images)

    print(f"\n--{name_data} EMBEDDING--\n")
    # Generate and load the embeddings for the test data
    assets = embedding_and_load_data(name_data, dir_processed_dataset_test, test_embed_dir, args, device)

    # Extract embeddings features for prediction
    X_test = assets["embeddings"]
    print("Embedding shape (X_test):", X_test.shape)

    print(f"\n--{name_data} REGRESSION PREDICTIONS--\n")
    # Make predictions and aggregate results across cross-validation regression models
    average_predictions = predict_and_aggregate_models(X_test, args.results_dir)

    ### Prepare and Return Predictions ###

    # Convert the predictions to a DataFrame (the gene expression value must be rounded to two decimal places)
    prediction = pd.DataFrame(np.round(average_predictions, 2), index=cell_ids, columns=gene_names)
    print("\n Predictions shape:", prediction.shape)

    ##############################################################################
    ##### BROAD Crunch 2: predict the expression of 2000 unmeasured genes #####
    print(f"\n** {name_data} Predict the expression of 2000 unmeasured genes (Crunch 2) ****************************************************************\n")

    print(f"--COSINE SIMILARITIES between {name_data} and single-cell RNA sequencing from shared genes--\n")

    # Read the scRNA-seq dataset and the 2000 unmeasured genes to predict
    scRNAseq = sc.read_h5ad(os.path.join(data_path, 'Crunch2_scRNAseq.h5ad'))
    gene_2000_list = pd.read_csv(os.path.join(data_path, 'Crunch2_gene_list.csv'))["gene_symbols"].tolist()

    # Extract shared and unmeasured genes
    gene_460 = list(prediction.columns)
    common_genes = [g for g in gene_460 if g in scRNAseq.var.index]
    print("Number of shared genes between scRNA-seq and xenium data:", len(common_genes))
    unmeasured_genes = [gene for gene in gene_2000_list if gene not in common_genes]
    print("Number of unmeasured genes in Xenium data:", len(unmeasured_genes), "\n")

    # there may be many other better ways to filter scRNAseq
    status = "Non-inflamed" if "NI" in name_data else 'Inflamed'  # status of tissue: ['Non-inflamed', 'Inflamed', 'Healthy', 'cancer']

    # Filter scRNA-seq on status to get only cells Seq with the specified status
    index_subset_status_filtered = scRNAseq.obs[scRNAseq.obs.status == status].index

    # For low-memory RAM system: sampling a subset of scRNAseq cells
    index_subset_status_filtered = index_subset_status_filtered[:10000]

    scRNAseq_status_filtered = scRNAseq[index_subset_status_filtered, common_genes + unmeasured_genes].copy()

    # scRNA-Seq data log1p-normalized with scale factor 10000 on 18615 genes
    rna_data_norm_10000 = scRNAseq_status_filtered.X.toarray()
    # scRNA-Seq data log1p-normalized with scale factor 100 on 460 genes
    rna_data_norm_100 = log1p_normalization_scale_factor(scRNAseq_status_filtered.layers["counts"].toarray(), scale_factor=100)

    del scRNAseq
    gc.collect()

    spot_vectors_df_scRNA_100 = pd.DataFrame(rna_data_norm_100, columns=list(scRNAseq_status_filtered.var.index))
    spot_vectors_df_scRNA_10000 = pd.DataFrame(rna_data_norm_10000, columns=list(scRNAseq_status_filtered.var.index))

    print(f"scRNA-Seq data shape ({len(rna_data_norm_10000)} samples x {spot_vectors_df_scRNA_100.shape[1]} shared genes + unmeasured genes)")
    print(f"Xenium data shape ({len(prediction)} samples x {len(common_genes)} shared genes)")

    # Similarity-Based Matching: Find the top_k most similar spots for each query
    top_k = 30
    print(f"\nCompute COSINE SIMILARITY: Find the top_k(={top_k}) similar scRNA-Seq cells for each Xenium cell...\n")
    matches, similarities = find_matches_cos_similarity(spot_vectors_df_scRNA_100[common_genes].values, prediction[common_genes].values, top_k=top_k)

    # Weighted Averaging of scRNA-Seq data log1p-normalized with scale factor 10000
    print("Compute WEIGHTED AVERAGE of unmeasured genes from scRNA-Seq based on similarity scores...")
    weighted_avg_df_10000 = pd.DataFrame([
        {
            **dict(zip(unmeasured_genes, np.average(spot_vectors_df_scRNA_10000.iloc[indices][unmeasured_genes].values, axis=0, weights=similarity).round(2)))
        }
        for i, (indices, similarity) in enumerate(zip(matches, similarities))
    ])
    weighted_avg_df_10000.index = prediction.index

    # Free memory by deleting large variables and performing garbage collection
    del average_predictions, prediction, sdata, X_test, assets, gene_2000_list, scRNAseq_status_filtered
    del rna_data_norm_10000, rna_data_norm_100, spot_vectors_df_scRNA_100, spot_vectors_df_scRNA_10000, matches, similarities
    gc.collect()

    # Apply log1p normalization and round to 2 decimal points
    weighted_avg_df_10000.iloc[:, :] = np.round(log1p_normalization_scale_factor(weighted_avg_df_10000.values, scale_factor=10000), 2)

    print(f"\n-- {name_data} PREDICTION DONE --\n")

    # Return the final prediction DataFrame
    return weighted_avg_df_10000

if __name__=="__main__":
    prediction = infer2(
        data_file_path="./data.2.large/UC9_I.zarr",
        model_directory_path="./resources"
    )
    prediction.to_csv('resources/crunch2.csv')
    print(prediction.head())

import os, torch, gc
from args1 import args1

def infer_crunch_1(
    name_data: str,  # The name of the dataset being processed (only used for logging and directory naming)
    data_file_path: str,
    model_directory_path: str,  # Path to save the trained model and results
    sdata: dict,  # a dataset as a dict containing at least an H&E image of tissue regions and a nuclear segmentation mask
    cell_ids: list,  # List of cell identifiers to predict gene expression
    gene_460_names: list,  # List of 460 gene names for which predictions will be made
):
    """
    Perform Crunch 1 inference: Predict the expression of 460 genes in spatial transcriptomics data
    using a pre-trained Resnet50 model and regression inference.
    """

    ### Prepare Directories ###
    print(f"\n-- {name_data} INFERENCE ---------------------------------------------------------------\n")

    # Previous Directory to store models and results
    dir_models_and_results = model_directory_path  # updated
    # Load training configuration parameters
    args, dir_processed_dataset, dir_models_and_results, list_ST_name_data = args1(model_directory_path)

    args.results_dir = dir_models_and_results  # updated

    # Directory for processed test dataset (temporary storage)
    dir_processed_dataset_test = os.path.join("resources", f"processed_dataset")
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

    # Convert the predictions to a DataFrame
    prediction_cell_ids_no_cancer = pd.DataFrame(np.round(average_predictions, 2), index=cell_ids, columns=gene_460_names)
    print("\n Predictions shape:", prediction_cell_ids_no_cancer.shape)

    del average_predictions, X_test, assets
    gc.collect()

    return prediction_cell_ids_no_cancer

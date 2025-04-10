"""
## Infer function

The `infer` function loads trained models and performs inference on a new dataset.

1. Prepare the necessary directories and load the configuration parameters from the previously trained model.
2. The test data, provided as a Zarr file, is read and specific subsets of the data (test and validation groups) are extracted.
3. Preprocess the data into image patches (X_test).
4. Generate embeddings for the test data and applies the trained models for regression predictions.
5. Format the predictions for submission.
"""
from types import SimpleNamespace
import joblib, os, json
from inf_encoder_factory import *
from generate_embeddings import *
from H5Dataset import *
import spatialdata as sd  # Manage multi-modal spatial omics datasets

def embedding_and_load_data(name_data, dir_processed_dataset_test, test_embed_dir, args, device):
    """
    Embedding of the images using the specified encoder and load the resulting data.

    Args:
    - name_data (str): The name of the data to process.
    - dir_processed_dataset_test (str): Directory where the processed test dataset is stored.
    - test_embed_dir (str): Directory where the embeddings should be saved.
    - args (namespace): Arguments object containing parameters like encoder, batch_size, etc.
    - device (torch.device): The device (CPU or GPU) to perform computations on.

    Returns:
    - assets (dict): Dictionary containing the 'barcodes', 'coords', and 'embeddings' from the embedded data.
    """

    # Print the encoder being used
    print(f"Embedding images using {args.encoder} encoder")

    # Create encoder based on the specified model and load its weights
    encoder = inf_encoder_factory(args.encoder)(args.weights_root)

    # Define the path for the patches data (h5 file)
    tile_h5_path = os.path.join(dir_processed_dataset_test, "patches", f'{name_data}.h5')

    # Check if the file exists
    assert os.path.isfile(tile_h5_path), f"Patches h5 file not found at {tile_h5_path}"

    # Define the embedding output path
    embed_path = os.path.join(test_embed_dir, f'{name_data}.h5')

    # Generate the embeddings and save them to the defined path
    generate_embeddings(embed_path, encoder, device, tile_h5_path, args.batch_size, args.num_workers, overwrite=args.overwrite)

    # Load the embeddings and related assets
    assets, _ = read_assets_from_h5(embed_path)

    # Extract cell IDs and convert to a list of strings
    # The cell IDs are not necessary because the images are kept in the same order as the gene expression data
    cell_ids = assets['barcodes'].flatten().astype(str).tolist()

    return assets


def load_models_from_directories(base_path):
    """
    Load all trained regression models (one model for each cross-validation split)
    Load 'model.pkl' from each directory within the base_path.

    :param base_path: The parent directory containing split subdirectories.
    :return: A dictionary where keys are directory names and values are the loaded models.
    """

    models = {}
    for name in os.listdir(base_path):
        dir_path = os.path.join(base_path, name)
        if os.path.isdir(dir_path):
            model_path = os.path.join(dir_path, 'model.pkl')
            if os.path.exists(model_path):
                models[name] = joblib.load(model_path)
                print(f"Loaded model from {model_path}")
            else:
                print(f"'model.pkl' not found in {dir_path}")

    return models


def predict_and_aggregate_models(X_test, results_dir):
    """
    Load models from the given directory, make predictions on the test set (X_test),
    aggregate the predictions by averaging, and set negative predictions to 0.

    Args:
    - X_test (np.array): The test data to make predictions on.
    - results_dir (str): Directory containing the saved models.

    Returns:
    - np.array: The aggregated predictions.
    """

    # Load models from the specified directory
    models = load_models_from_directories(results_dir)

    # Initialize a list to store predictions
    predictions = []

    # Iterate through each model and make predictions
    for split_name in models.keys():
        preds = models[split_name].predict(X_test)
        predictions.append(preds)

    # Stack the predictions into a 2D array (models x samples)
    predictions = np.stack(predictions)

    # Aggregate predictions by calculating the mean across all models
    average_predictions = np.mean(predictions, axis=0)

    # Set any negative predictions to 0
    average_predictions = np.where(average_predictions < 0, 0.0, average_predictions)

    del models

    return average_predictions

# In the inference function, the trained model is loaded and used to make inferences on a
# sample of data that matches the characteristics of the training test.
def infer1(
    data_file_path: str,  # Path to a test dataset (in Zarr format) to perform inference on.
    model_directory_path: str  # Path to save the trained model and results
):
    ### Prepare Directories ###

    # Extract the name of the dataset from the file path (without extension)
    name_data = os.path.splitext(os.path.basename(data_file_path))[0]
    print(f"\n-- {name_data} INFERENCE ---------------------------------------------------------------\n")
    print(data_file_path)

    # Previous directory where models and results are stored
    dir_models_and_results = os.path.join(model_directory_path, f"ST_pred_results")
    # Load training configuration parameters
    config_path = os.path.join(dir_models_and_results, "config.json")
    with open(config_path, 'r') as f:
        args_dict = json.load(f)
    args = SimpleNamespace(**args_dict)

    # Directory for processed test dataset (temporary storage)
    dir_processed_dataset_test = os.path.join("resources", f"processed_dataset_test")
    os.makedirs(dir_processed_dataset_test, exist_ok=True)

    # Directory to store the test data embeddings (temporary storage)
    test_embed_dir = os.path.join(dir_processed_dataset_test, "ST_data_emb")
    os.makedirs(test_embed_dir, exist_ok=True)

    # Set device to GPU if available, else use CPU!!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    print(f"\n-- {name_data} EMBEDDING--\n")
    # Generate and load the embeddings for the test data
    assets = embedding_and_load_data(name_data, dir_processed_dataset_test, test_embed_dir, args, device)

    # Extract embeddings features for prediction
    X_test = assets["embeddings"]
    print("Embedding shape (X_test):", X_test.shape)

    print(f"\n-- {name_data} REGRESSION PREDICTIONS--\n")
    # Make predictions and aggregate results across cross-validation regression models
    average_predictions = predict_and_aggregate_models(X_test, args.results_dir)

    ### Prepare and Return Predictions ###

    # Convert the predictions to a DataFrame (the gene expression value must be rounded to two decimal places)
    prediction = pd.DataFrame(np.round(average_predictions, 2), index=cell_ids, columns=gene_names)
    # Reset index to have 'cell_id' as a column
    prediction = prediction.reset_index(names="cell_id")

    # Melt the DataFrame to the expected output for the challenge
    prediction = prediction.melt(id_vars="cell_id", var_name="gene", value_name="prediction")
    # prediction = prediction.sort_values(by=["cell_id", "gene"]).reset_index(drop=True)

    # Free memory by deleting large variables and performing garbage collection
    del average_predictions, sdata, X_test, assets
    gc.collect()

    print(f"\n-- {name_data} PREDICTION DONE\n")

    # Return the final prediction DataFrame
    return prediction
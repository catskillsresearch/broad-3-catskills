import os
from types import SimpleNamespace

def args1(model_directory_path):

    os.makedirs(model_directory_path, exist_ok=True)

    # Directory to store the processed train dataset (temporary storage)
    dir_processed_dataset = os.path.join("resources", f"processed_dataset")
    os.makedirs(dir_processed_dataset, exist_ok=True)

    # Directory to store models and results
    dir_models_and_results = os.path.join(model_directory_path, f"ST_pred_results")
    os.makedirs(dir_models_and_results, exist_ok=True)

    # List of datasets for Spatial Transcriptomics (ST) training
    # The DC1 sample is deliberately absent because it does not contain spatial transcriptomic data.
    #list_ST_name_data = ["UC1_NI", "UC1_I", "UC6_NI", "UC6_I", "UC7_I", "UC9_I", "DC5"]
    list_ST_name_data = ["UC1_I", "UC9_I"]
    
    # Training parameters
    args_dict = {
        # Parameters for data preprocessing
        "size_subset": None, # 10000,  # Sampling 10,000 smaller images from each H&E image
        "target_patch_size": 32,  # Target size of cell patches (sub-region of an image) for the data

        "show_extracted_images": True,  # (Only for visualization) Whether to visualize all the extracted patches for the first ST data
        "vis_width": 1000,  # (Only for visualization) Width of the above visualization

        # Generic training settings
        "seed": 1,  # Random seed for reproducibility
        "overwrite": False,  # Whether to overwrite the existing embedding files (Set to True if you have just changed some data preprocessing parameters)
        "dir_dataset": dir_processed_dataset,  # Path to save the processed train dataset
        "embed_dataroot": os.path.join(dir_processed_dataset, f"ST_data_emb"),  # Path for embedding data
        "results_dir": dir_models_and_results,  # Directory to save results
        "n_fold": 2,  # Number of folds for leave-one-out cross-validation

        # Encoder settings
        "batch_size": 128,  # Batch size for training
        "num_workers": 0,  # Number of workers for loading data (set 0 for submission on CrunchDAO plaform)

        # Train settings
        "gene_list": "var_genes.json",  # Path to save the list of genes
        "method": "ridge",  # Regression method to use
        "alpha": None,  # Regularization parameter for ridge regression
        "normalize": False,  # Whether to normalize the data (sdata["anucleus"].X is already normalized with scale factor 100)
        "encoder": "resnet50",  # Encoder model to use (e.g., ResNet50)
        "weights_root": os.path.join(model_directory_path, f"pytorch_model.bin")  # Path to the pre-trained model weights
    }

    # Convert the dictionary to a simple namespace for easy access to attributes
    args = SimpleNamespace(**args_dict)
    return args, dir_processed_dataset, dir_models_and_results, list_ST_name_data
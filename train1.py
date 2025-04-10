"""

Train1 function

The `train1` function automates the training of models for spatial transcriptomics data.

1. Prepare the input data by preprocessing the spatial transcriptomics datasets (image patches (X) and gene expression (Y)).
2. Create leave-one-out cross-validation splits for training and testing.
3. Use the specified encoder model (ResNet50) and a regression method (ridge regression) to train the model
4. Save the results and the trained model in the `.resources/` directories for later inference.

![train_architecture](https://raw.githubusercontent.com/crunchdao/competitions/refs/heads/master/competitions/broad-1/quickstarters/resnet50-plus-ridge/images/train_architecture.png)

`Y` is log1p-normalized with scale factor 100.

All regression models and metric results are saved in `./resources/ST_pred_results` -- check it out!

Preprocessed datasets are saved in `resources/processed_dataset` (temporary storage - not needed for inference).

Official challenge metric: L2 mean error (`l2_errors_mean`)
"""

import os
from extract_spatial_positions import *
from run_training import *
from args1 import args1

# In the training function, users build and train the model to make inferences on the test data.
# Your train models must be stored in the `model_directory_path`.
def train1(
    data_directory_path: str, # Path to the input data directory
    model_directory_path: str # Path to save the trained model and results
):
    print("\n-- TRAINING ---------------------------------------------------------------\n")

    args, dir_processed_dataset, dir_models_and_results, list_ST_name_data = args1(model_directory_path)

    # Preprocess the spatial transcriptomics data for each sample in list_ST_name_data -> all preprocessed data saved in dir_processed_dataset
    preprocess_spatial_transcriptomics_data_train(list_ST_name_data, data_directory_path, dir_processed_dataset,
                                                  args.size_subset, args.target_patch_size, args.vis_width, args.show_extracted_images)

    # Create leave-one-out cross-validation splits for training and testing in csv files
    create_cross_validation_splits(dir_processed_dataset, n_fold=args.n_fold)

    # Run the training process using the parameters set above
    run_training(args)

    # View all results in directory `args.results_dir`
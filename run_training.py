"""

Training Functions

This section defines functions for training models and saving results.

- **`predict_single_split`**: Performs predictions for a single train-test split. It handles the embedding generation, loading of gene expression data, model training and saving the results. The trained model is saved to a specified directory.
- **`predict_folds`**: Performs predictions for all train-test folds, iterating over the splits and calling `predict_single_split` for each fold. It saves the results for all folds, including a summary of Pearson correlations, L2 errors and R² scores.
- **`run_training`**: Executes the full training process by calling `predict_folds` and saving the encoder performance results.
"""

import os, json
import pandas as pd
from inf_encoder_factory import *
from generate_embeddings import *
from H5Dataset import *
from train_test_reg import *
from tqdm import tqdm
from utilities import *
import joblib

def predict_single_split(fold, train_split, test_split, args, save_dir, model_name, device, bench_data_root):
    """ Predict a single split for a single model """

    # Ensure paths to train and test split files are correct
    if not os.path.isfile(train_split):
        train_split = os.path.join(bench_data_root, 'splits', train_split)

    if not os.path.isfile(test_split):
        test_split = os.path.join(bench_data_root, 'splits', test_split)

    # Read train and test split CSV files
    train_df = pd.read_csv(train_split)
    test_df = pd.read_csv(test_split)

    # Directory to save embedding results
    embedding_dir = args.embed_dataroot
    os.makedirs(embedding_dir, exist_ok=True)

    # Embedding process
    print("\n--EMBEDDING--\n")
    print(f"Embedding tiles using {model_name} encoder")
    encoder: InferenceEncoder = inf_encoder_factory(model_name)(args.weights_root)

    # Loop over train and test splits to generate embeddings
    for split in [train_df, test_df]:
        for i in range(len(split)):
            sample_id = split.iloc[i]['sample_id']
            tile_h5_path = os.path.join(bench_data_root, split.iloc[i]['patches_path'])
            assert os.path.isfile(tile_h5_path)
            embed_path = os.path.join(embedding_dir, f'{sample_id}.h5')
            print(f"\nGENERATE EMBEDDING - {sample_id}\n")
            generate_embeddings(embed_path, encoder, device, tile_h5_path, args.batch_size, args.num_workers, overwrite=args.overwrite)

    # Initialize dictionary to hold data for both splits
    all_split_assets = {}

    gene_list = args.gene_list

    # print(f'using gene_list {gene_list}')
    # Load gene list for expression data
    with open(os.path.join(bench_data_root, gene_list), 'r') as f:
        genes = json.load(f)['genes']

    # Process train and test splits
    for split_key, split in zip(['train', 'test'], [train_df, test_df]):
        split_assets = {}
        for i in tqdm(range(len(split))):
            # Get sample ID, embedding path and gene expression path
            sample_id = split.iloc[i]['sample_id']
            embed_path = os.path.join(embedding_dir, f'{sample_id}.h5')
            expr_path = os.path.join(bench_data_root, split.iloc[i]['expr_path'])

            # Read embedding data and gene expression data
            assets, _ = read_assets_from_h5(embed_path)
            barcodes = assets['barcodes'].flatten().astype(str).tolist()
            adata = load_adata(expr_path, genes=genes, barcodes=barcodes, normalize=args.normalize)
            assets['adata'] = adata.values

            # Merge assets for the split
            split_assets = merge_dict(split_assets, assets)

        # Concatenate all data in the split
        for key, val in split_assets.items():
            split_assets[key] = np.concatenate(val, axis=0)

        all_split_assets[split_key] = split_assets
        print(f"Loaded {split_key} split with {len(split_assets['embeddings'])} samples: {split_assets['embeddings'].shape}")

    # Assign data to X and y variables for training and testing
    X_train, y_train = all_split_assets['train']['embeddings'], all_split_assets['train']['adata']
    X_test, y_test = all_split_assets['test']['embeddings'], all_split_assets['test']['adata']

    print("\n--REGRESSION--\n")
    # Perform regression using the specified method
    reg, probe_results = train_test_reg(X_train, X_test, y_train, y_test, random_state=args.seed, genes=genes, method=args.method)

    # Save the trained regression model
    model_path = os.path.join(save_dir, f'model.pkl')
    joblib.dump(reg, model_path)
    print(f"Model saved in '{model_path}'")

    # Summarize results for the current fold
    print(f"\n--FOLD {fold} RESULTS--\n")
    probe_summary = {}
    probe_summary.update({'n_train': len(y_train), 'n_test': len(y_test)})
    probe_summary.update({key: val for key, val in probe_results.items()})
    keys_to_print = ["n_train", "n_test", "pearson_mean", "l2_errors_mean", "r2_scores_mean", "l2_error_q1", "l2_error_q2", "l2_error_q3"]

    filtered_summary = {
        key: round(probe_summary[key], 4)
        for key in keys_to_print
        if key in probe_summary
    }
    print(filtered_summary)

    with open(os.path.join(save_dir, f'results.json'), 'w') as f:
        json.dump(probe_results, f, sort_keys=True, indent=4)

    with open(os.path.join(save_dir, f'summary.json'), 'w') as f:
        json.dump(probe_summary, f, sort_keys=True, indent=4)

    return probe_results


def predict_folds(args, exp_save_dir, model_name, device, bench_data_root):
    """ Predict all folds for a given model """

    # Define the directory for splits
    split_dir = os.path.join(bench_data_root, 'splits')
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"{split_dir} doesn't exist, make sure that you specified the splits directory")

    splits = os.listdir(split_dir)
    if len(splits) == 0:
        raise FileNotFoundError(f"{split_dir} is empty, make sure that you specified train and test files")

    n_splits = len(splits) // 2

    # Save training configuration to JSON
    with open(os.path.join(exp_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    # Loop through each split and perform predictions
    libprobe_results_arr = []
    for i in range(n_splits):
        print(f"\n--FOLD {i}--\n")
        train_split = os.path.join(split_dir, f'train_{i}.csv')
        test_split = os.path.join(split_dir, f'test_{i}.csv')
        kfold_save_dir = os.path.join(exp_save_dir, f'split{i}')
        os.makedirs(kfold_save_dir, exist_ok=True)

        # Predict using the current fold
        linprobe_results = predict_single_split(i, train_split, test_split, args, kfold_save_dir, model_name, device=device, bench_data_root=bench_data_root)
        libprobe_results_arr.append(linprobe_results)

    # Merge and save k-fold results
    kfold_results = merge_fold_results(libprobe_results_arr)
    with open(os.path.join(exp_save_dir, f'results_kfold.json'), 'w') as f:
        p_corrs = kfold_results['pearson_corrs']
        p_corrs = sorted(p_corrs, key=itemgetter('mean'), reverse=True)
        kfold_results['pearson_corrs'] = p_corrs
        json.dump(kfold_results, f, sort_keys=True, indent=4)

    return kfold_results


def run_training(args):
    """ Training function: Execute predict_folds for processed train dataset with the specified encoder and dump the results in a nested directory structure """

    print("\n-- RUN TRAINING ---------------------------------------------------------------\n")

    print(f'run parameters {args}')
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create directory for results
    save_dir = args.results_dir
    os.makedirs(save_dir, exist_ok=True)

    # Perform predictions for all folds
    enc_results = predict_folds(args, save_dir, model_name=args.encoder, device=device, bench_data_root=args.dir_dataset)

    # Store and save encoder performance results
    enc_perfs = {
        'encoder_name': args.encoder,
        'pearson_mean': round(enc_results['pearson_mean'], 4),
        'pearson_std': round(enc_results['pearson_std'], 4),
    }

    with open(os.path.join(save_dir, 'dataset_results.json'), 'w') as f:
        json.dump(enc_perfs, f, sort_keys=True, indent=4)

    print("\n-- TRAINING DONE ---------------------------------------------------------------\n")

    print("\n-- Leave-one-out CV performance ---------------------------------------------------------------")
    print(enc_perfs)
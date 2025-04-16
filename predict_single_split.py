

from operator import itemgetter

def predict_single_split(fold, train_split, test_split, args, save_dir, model_name, device, bench_data_root):
    """**`predict_single_split`**:
    	Performs predictions for a single train-test split.
    	It handles the embedding generation, loading of gene expression data, model training and saving the results.
    	The trained model is saved to a specified directory."""

    # Ensure paths to train and test split files are correct
    import os
    if not os.path.isfile(train_split):
        train_split = os.path.join(bench_data_root, 'splits', train_split)

    print("train_split", train_split, os.path.isfile(train_split))

    if not os.path.isfile(test_split):
        test_split = os.path.join(bench_data_root, 'splits', test_split)

    # Read train and test split CSV files
    import pandas as pd
    train_df = pd.read_csv(train_split)
    test_df = pd.read_csv(test_split)

    # Directory to save embedding results
    embedding_dir = args.embed_dataroot
    os.makedirs(embedding_dir, exist_ok=True)

    # Embedding process
    print("\n--EMBEDDING--\n")
    print(f"Embedding tiles using {model_name} encoder")
    from inf_encoder_factory import inf_encoder_factory
    encoder: InferenceEncoder = inf_encoder_factory(model_name)(args.weights_root)

    # Loop over train and test splits to generate embeddings
    for split in [train_df, test_df]:
        for i in range(len(split)):
            sample_id = split.iloc[i]['sample_id']
            tile_h5_path = os.path.join(bench_data_root, split.iloc[i]['patches_path'])
            assert os.path.isfile(tile_h5_path)
            embed_path = os.path.join(embedding_dir, f'{sample_id}.h5')
            print(f"\nGENERATE EMBEDDING - {sample_id}\n")
            from generate_embeddings import generate_embeddings
            generate_embeddings(embed_path, encoder, device, tile_h5_path, args.batch_size, args.num_workers, overwrite=args.overwrite)

    # Initialize dictionary to hold data for both splits
    all_split_assets = {}

    import json
    gene_list = args.gene_list
    genes_path = os.path.join(bench_data_root, gene_list)
    with open(os.path.join(bench_data_root, gene_list), 'r') as f:
        genes = json.load(f)['genes']

    # Process train and test splits
    import numpy as np
    from tqdm import tqdm
    from load_adata import load_adata
    from read_assets_from_h5 import read_assets_from_h5
    from merge_dict import merge_dict
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
    X_train_fn = os.path.join(save_dir, 'X_train.npy')
    y_train_fn = os.path.join(save_dir, 'y_train.npy')
    X_test_fn = os.path.join(save_dir, 'X_test.npy')
    y_test_fn = os.path.join(save_dir, 'y_test.npy')
    np.save(X_train_fn, X_train)
    np.save(y_train_fn, y_train)
    np.save(X_test_fn, X_test)
    np.save(y_test_fn, y_test)
    
    print("\n--REGRESSION--\n")
    model_path = os.path.join(save_dir, f'model.pkl')
    probe_results_fn = os.path.join(save_dir, f'results.json')
    if not os.path.exists(probe_results_fn):
        # Perform regression using the specified method
        import subprocess
        pt = '/home/catskills/Desktop/broad/broad_cudf/bin/python'
        fn = 'train_test_reg.py'
        cmd = [pt, fn, model_path, genes_path, probe_results_fn, 
            X_train_fn, X_test_fn, y_train_fn, y_test_fn,
            str(args.seed), args.method]
        print("cmd: ", ' '.join(cmd))
        subprocess.run(cmd)
    import json
    with open(probe_results_fn, 'r') as f:
        probe_results = json.load(f)
        
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

    from NpEncoder import NpEncoder
    with open(os.path.join(save_dir, f'summary.json'), 'w') as f:
        json.dump(probe_summary, f, sort_keys=True, indent=4, cls=NpEncoder)

    return probe_results

""" **`predict_folds`**:
        Performs predictions for all train-test folds, iterating over the splits and calling `predict_single_split` for each fold.
	It saves the results for all folds, including a summary of Pearson correlations, L2 errors and R² scores.
"""

def predict_folds(args, exp_save_dir, model_name, device, bench_data_root):
    """ Predict all folds for a given model """

    # Define the directory for splits
    import os
    split_dir = os.path.join(bench_data_root, 'splits')
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"{split_dir} doesn't exist, make sure that you specified the splits directory")

    splits = os.listdir(split_dir)
    if len(splits) == 0:
        raise FileNotFoundError(f"{split_dir} is empty, make sure that you specified train and test files")

    n_splits = len(splits) // 2

    # Save training configuration to JSON
    from NpEncoder import NpEncoder
    import json
    with open(os.path.join(exp_save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4, cls=NpEncoder)

    # Loop through each split and perform predictions
    libprobe_results_arr = []
    for i in range(n_splits):
        print(f"\n--FOLD {i}--\n")
        train_split = os.path.join(split_dir, f'train_{i}.csv')
        test_split = os.path.join(split_dir, f'test_{i}.csv')
        kfold_save_dir = os.path.join(exp_save_dir, f'split{i}')
        os.makedirs(kfold_save_dir, exist_ok=True)

        # Predict using the current fold
        from predict_single_split import predict_single_split
        linprobe_results = predict_single_split(i, train_split, test_split, args, kfold_save_dir, model_name,
                                                device=device, bench_data_root=bench_data_root)
        libprobe_results_arr.append(linprobe_results)

    # Merge and save k-fold results
    from operator import itemgetter
    from merge_fold_results import merge_fold_results
    kfold_results = merge_fold_results(libprobe_results_arr)
    with open(os.path.join(exp_save_dir, f'results_kfold.json'), 'w') as f:
        p_corrs = kfold_results['pearson_corrs']
        p_corrs = sorted(p_corrs, key=itemgetter('mean'), reverse=True)
        kfold_results['pearson_corrs'] = p_corrs
        json.dump(kfold_results, f, sort_keys=True, indent=4, cls=NpEncoder)

    return kfold_results

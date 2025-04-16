def run_training(args):
    """ Training function: Execute predict_folds for processed train dataset with the specified encoder and dump the results in a nested directory structure """
    save_dir = args.results_dir

    import os
    enc_perf_fn = os.path.join(save_dir, 'dataset_results.json')
    if os.path.exists(enc_perf_fn):
        return
        
    print("\n-- RUN TRAINING ---------------------------------------------------------------\n")

    print(f'run parameters {args}')
    # Set device to GPU if available
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create directory for results
    os.makedirs(save_dir, exist_ok=True)

    # Perform predictions for all folds
    from predict_folds import predict_folds
    enc_results = predict_folds(args, save_dir, model_name=args.encoder, device=device, bench_data_root=args.dir_dataset)

    # Store and save encoder performance results
    enc_perfs = {
        'encoder_name': args.encoder,
        'pearson_mean': round(enc_results['pearson_mean'], 4),
        'pearson_std': round(enc_results['pearson_std'], 4),
    }

    import json
    with open(enc_perf_fn, 'w') as f:
        json.dump(enc_perfs, f, sort_keys=True, indent=4, cls=NpEncoder)

    print("\n-- TRAINING DONE ---------------------------------------------------------------\n")

    print("\n-- Leave-one-out CV performance ---------------------------------------------------------------")
    print(enc_perfs)

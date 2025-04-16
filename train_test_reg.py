def train_test_reg(model_path, genes_path, results_fn, X_train_fn, X_test_fn, y_train_fn, y_test_fn,
                   random_state=0, method='ridge'):
    """ Train a regression model using cuML and evaluate its performance on test data """
    
    # Load gene list for expression data
    import json
    with open(genes_path, 'r') as f:
        genes = json.load(f)['genes']

    import numpy as np
    X_train = np.load(X_train_fn).astype(np.float32)
    X_test = np.load(X_test_fn).astype(np.float32)
    Y_train = np.load(y_train_fn).astype(np.float32)
    Y_test = np.load(y_test_fn).astype(np.float32)

    if method == 'ridge':
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        Y_test_t = torch.tensor(Y_test, dtype=torch.float32, device=device)

        from ridge_regression import ridge_regression
        alpha = 100 / (X_train.shape[1] * Y_train.shape[1])
        print(f"Ridge: using alpha: {alpha}")
        W = ridge_regression(X_train_t, Y_train_t, alpha=alpha)

        # Make predictions on test set
        with torch.no_grad():
            Y_pred_t = X_test_t @ W
        Y_pred = Y_pred_t.cpu().numpy()
        
    # Compute the evaluation metrics using the test data and predictions
    from compute_metrics import compute_metrics
    results = compute_metrics(Y_test, Y_pred, genes=genes)

    # Quick MSE from SKLearn
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(Y_test, Y_pred)
    print("MSE", mse)
    
    # Save the trained regression model
    import joblib
    joblib.dump(W, model_path)
    print(f"Model saved in '{model_path}'")
    from NpEncoder import NpEncoder
    with open(results_fn, 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4, cls=NpEncoder)
    print(f'Results saved in {results_fn}')
    
if __name__=="__main__":
    import sys
    if 0:
        model_path, genes_path, results_fn, X_train_fn, X_test_fn, y_train_fn, y_test_fn, random_state, method = sys.argv[1:]
    else:
        model_path, genes_path, results_fn, X_train_fn, X_test_fn, y_train_fn, y_test_fn, random_state, method = './resources/ST_pred_results/split0/model.pkl resources/processed_dataset/var_genes.json ./resources/ST_pred_results/split0/results.json ./resources/ST_pred_results/split0/X_train.npy ./resources/ST_pred_results/split0/X_test.npy ./resources/ST_pred_results/split0/y_train.npy ./resources/ST_pred_results/split0/y_test.npy 1 ridge'.split(' ')
    random_state = int(random_state)
    train_test_reg(model_path, genes_path, results_fn, X_train_fn, X_test_fn, y_train_fn, y_test_fn, random_state, method)

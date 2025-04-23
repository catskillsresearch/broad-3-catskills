import torch

def ridge_regression(X, Y, alpha=1.0):
    """
    Closed-form solution for multi-output Ridge regression
    X: (n_samples, n_features) tensor
    Y: (n_samples, n_targets) tensor
    Returns: (n_features, n_targets) weight matrix
    """
    # Add bias term (optional, remove if not needed)
    # X = torch.cat([X, torch.ones(X.shape[0], 1, device=device)], dim=1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Compute closed-form solution
    XtX = X.T @ X
    regularization = alpha * torch.eye(XtX.shape[0], device=device)
    weights = torch.linalg.solve(XtX + regularization, X.T @ Y)
    return weights

def ridge_fit(X, Y):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train_t = torch.tensor(X, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y, dtype=torch.float32, device=device)
    
    # Train the model
    alpha = 100 / (X_train_t.shape[1] * Y_train_t.shape[1])
    
    W_t = ridge_regression(X_train_t, Y_train_t, alpha=alpha)

    return W_t.cpu().numpy()

def ridge_apply(X, W):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    W_t = torch.tensor(W, dtype=torch.float32, device=device)

    # Make predictions on test set
    with torch.no_grad():
        Y_t = X_t @ W_t

    return Y_t.cpu().numpy()

if __name__=="__main__":
    import torch
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Generate synthetic multi-output regression data
    n_samples = 10000
    n_features = 100
    n_targets = 5  # Number of output dimensions
    
    X, Y = make_regression(n_samples=n_samples,
                           n_features=n_features,
                           n_targets=n_targets,
                           noise=0.1,
                           random_state=42)
    
    # Split into train/test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors and move to GPU
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    # Train the model
    alpha = 100 / (X_train.shape[1] * Y_train.shape[1])
    W = ridge_regression(X_train_t, Y_train_t, alpha=alpha)
    # Make predictions on test set
    with torch.no_grad():
        Y_pred_t = X_test_t @ W
    # Move predictions back to CPU for evaluation with sklearn
    Y_pred = Y_pred_t.cpu().numpy()
    # Evaluate
    mse = mean_squared_error(Y_test, Y_pred)
    print(f"Test MSE: {mse:.4f} for alpha: {alpha}")
    print(f"Weight matrix shape: {W.shape} (features × targets)")

    # Comparison
    from sklearn.linear_model import Ridge  # Regression model
    max_iter=1000
    random_state=0

    print(f"Scikit-Learn Ridge: using alpha: {alpha} versus calc alpha {alpha}")
    # Initialize Ridge regression model
    reg = Ridge(solver='lsqr',
                alpha=alpha,
                random_state=random_state,
                fit_intercept=False,
                max_iter=max_iter)
    reg.fit(X_train, Y_train)

    # Make predictions on the test data
    Y_pred1 = reg.predict(X_test)
    mse = mean_squared_error(Y_test, Y_pred1)
    print(f"Test MSE: {mse:.4f}")

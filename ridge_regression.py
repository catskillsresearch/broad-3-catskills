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
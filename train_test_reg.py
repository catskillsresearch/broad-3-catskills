"""

Regression Functions

This section defines functions for training a regression model and evaluating its performance.

* **`compute_metrics`**: Computes evaluation metrics, including L2 error, R² score and Pearson correlation for each target/gene.
* **`train_test_reg`**: Trains a regression model (default: Ridge regression) using the provided embedding vector train data and evaluates it on test data.
"""

import warnings
import numpy as np
from scipy.stats import pearsonr, ConstantInputWarning
from sklearn.linear_model import Ridge  # Regression model

def compute_metrics(y_test, preds_all, genes=None):
    """
    Computes metrics L2 errors R2 scores and Pearson correlations for each target/gene.

    :param y_test: Ground truth values (numpy array of shape [n_samples, n_targets]).
    :param preds_all: Predictions for all targets (numpy array of shape [n_samples, n_targets]).
    :param genes: Optional list of gene names corresponding to targets.
    :return: A dictionary containing metrics.
    """

    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []

    for i, target in enumerate(range(y_test.shape[1])):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]

        # Compute L2 error
        l2_error = float(np.mean((preds - target_vals) ** 2))

        # Compute R2 score
        total_variance = np.sum((target_vals - np.mean(target_vals)) ** 2)
        if total_variance == 0:
            r2_score = 0.0
        else:
            r2_score = float(1 - np.sum((target_vals - preds) ** 2) / total_variance)

        # Compute Pearson correlation
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConstantInputWarning)
            try:
                pearson_corr, _ = pearsonr(target_vals, preds)
                pearson_corr = pearson_corr if not np.isnan(pearson_corr) else 0.0
            except ConstantInputWarning:
                pearson_corr = 0.0

        errors.append(l2_error)
        r2_scores.append(r2_score)
        pearson_corrs.append(pearson_corr)

        # Record gene-specific Pearson correlation
        if genes is not None:
            pearson_genes.append({
                'name': genes[i],
                'pearson_corr': pearson_corr
            })

    # Compile results
    results = {
        'pearson_mean': float(np.mean(pearson_corrs)),
        'l2_errors_mean': float(np.mean(errors)),
        'r2_scores_mean': float(np.mean(r2_scores)),
        'l2_errors': list(errors),
        'r2_scores': list(r2_scores),
        'pearson_corrs': pearson_genes if genes is not None else list(pearson_corrs),
        'pearson_std': float(np.std(pearson_corrs)),
        'l2_error_q1': float(np.percentile(errors, 25)),
        'l2_error_q2': float(np.median(errors)),
        'l2_error_q3': float(np.percentile(errors, 75)),
        'r2_score_q1': float(np.percentile(r2_scores, 25)),
        'r2_score_q2': float(np.median(r2_scores)),
        'r2_score_q3': float(np.percentile(r2_scores, 75)),
    }

    return results


def train_test_reg(X_train, X_test, y_train, y_test,
                   max_iter=1000, random_state=0, genes=None, alpha=None, method='ridge'):
    """ Train a regression model and evaluate its performance on test data """

    if method == 'ridge':
        # If alpha is not provided, compute it based on the input dimensions
        alpha = 100 / (X_train.shape[1] * y_train.shape[1])

        print(f"Ridge: using alpha: {alpha}")
        # Initialize Ridge regression model
        reg = Ridge(solver='lsqr',
                    alpha=alpha,
                    random_state=random_state,
                    fit_intercept=False,
                    max_iter=max_iter)
        # Fit the model on the training data
        reg.fit(X_train, y_train)

        # Make predictions on the test data
        preds_all = reg.predict(X_test)

    # You can instantiate other regression models...

    # Compute the evaluation metrics using the test data and predictions
    results = compute_metrics(y_test, preds_all, genes=genes)

    return reg, results
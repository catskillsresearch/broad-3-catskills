import numpy as np

def log1p_normalization_scale_factor(arr, scale_factor=10000):
    row_sums = np.sum(arr, axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
    return np.log1p((arr / row_sums) * scale_factor)
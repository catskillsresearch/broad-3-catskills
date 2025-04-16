import scanpy as sc  # For analyzing single-cell data, especially for dimensionality reduction and clustering.
import numpy as np

def log1p_normalization(arr):
    """  Apply log1p normalization to the given array """

    scale_factor = 100
    return np.log1p((arr / np.sum(arr, axis=1, keepdims=True)) * scale_factor)

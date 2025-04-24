import numpy as np

def select_random_from_2D_array(arr, size):
    # Step 1: Get non-zero indices
    rows, cols = np.nonzero(arr)
    
    # Step 2: Sample 10,000 indices
    random_indices = np.random.choice(len(rows), size=size, replace=False)
    
    # Step 3: Extract elements
    sampled_elements = arr[rows[random_indices], cols[random_indices]]
    
    return sampled_elements

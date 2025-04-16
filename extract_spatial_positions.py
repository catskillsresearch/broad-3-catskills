"""
Preprocessing Spatial Transcriptomics Data

This section contains functions for preprocessing spatial transcriptomics data, including extracting spatial coordinates, generating image patches and preparing datasets for training and testing.

* **`extract_spatial_positions`**: Extracts spatial coordinates (centroids) of cells.
* **`process_and_visualize_image`**: Extracts square image patches from H&E images and visualizes them.
* **`preprocess_spatial_transcriptomics_data_train`**: Prepares training data by generating gene expression (Y) and image patch datasets (X).
* **`preprocess_spatial_transcriptomics_data_test`**: Prepares test data by generating image patches (X) for selected cells.
* **`create_cross_validation_splits`**: Creates leave-one-out cross-validation splits for model evaluation.

![data_X_Y](https://raw.githubusercontent.com/crunchdao/competitions/refs/heads/master/competitions/broad-1/quickstarters/resnet50-plus-ridge/images/data_X_Y.png)

`Leave-one-out cross-validation schema:`
![cross_validation](https://raw.githubusercontent.com/crunchdao/competitions/refs/heads/master/competitions/broad-1/quickstarters/resnet50-plus-ridge/images/cross_validation.png)
"""

import os
import spatialdata as sd  # Manage multi-modal spatial omics datasets
from skimage.measure import regionprops  # Get region properties of nucleus/cell image from masked nucleus image
from tqdm import tqdm
import numpy as np
import pandas as pd
from save_hdf5 import *
import gc
import json
import warnings

def extract_spatial_positions(sdata, cell_id_list):
    """
    Extracts spatial positions (centroids) of regions from the nucleus image where cell IDs match the provided cell list.

    Need to use 'HE_nuc_original' to extract spatial coordinate of cells
    HE_nuc_original: The nucleus segmentation mask of H&E image, in H&E native coordinate system. 
                     The cell_id in this segmentation mask matches with the nuclei by gene matrix stored in anucleus.
    HE_nuc_original is like a binary segmentation mask 0 - 1 but replace 1 with cell_ids.
    You can directly find the location of a cell, with cell_id, through HE_nuc_original==cell_id

    Parameters:
    -----------
    sdata: SpatialData
        A spatial data object containing the nucleus segmentation mask ('HE_nuc_original').
    cell_id_list: array-like
        A list or array of cell IDs to filter the regions.

    Returns:
    --------
    np.ndarray
        A NumPy array of spatial coordinates (x_center, y_center) for matched regions.
    """

    print("Extracting spatial positions ...")
    # Get region properties from the nucleus image: for each cell_id get its location on HE image
    if "tif_HE_nuc" in sdata:
        regions = regionprops(sdata['tif_HE_nuc'])
    else:
        regions = regionprops(sdata['HE_nuc_original'][0, :, :].to_numpy())

    dict_spatial_positions = {}
    # Loop through each region and extract centroid if the cell ID matches
    for props in tqdm(regions):
        cell_id = props.label
        centroid = props.centroid
        # Extract only coordinates from the provided cell_id list
        if cell_id in cell_id_list:
            y_center, x_center = int(centroid[0]), int(centroid[1])
            dict_spatial_positions[cell_id] = [x_center, y_center]

    # To maintain cell IDs order
    spatial_positions = []
    for cell_id in cell_id_list:
        try:
            spatial_positions.append(dict_spatial_positions[cell_id])
        except KeyError:
            print(f"Warning: Cell ID {cell_id} not found in the segmentation mask.")
            spatial_positions.append([1000, 1000])

    return np.array(spatial_positions)

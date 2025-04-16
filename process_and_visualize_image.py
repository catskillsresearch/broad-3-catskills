
def process_and_visualize_image(sdata, patch_save_dir, name_data, coords_center, target_patch_size, barcodes,
                                show_extracted_images=True, vis_width=1000, dpi=150):
    """
    Load and process the spatial image data, creates patches, saves them in an HDF5 file,
    and visualizes the extracted images and spatial coordinates.

    Parameters:
    -----------
    sdata: SpatialData
        A spatial data object containing the image to process ('HE_original') and associated metadata.
    patch_save_dir: str
        Directory where the resulting HDF5 file and visualizations will be saved.
    name_data: str
        Name used for saving the dataset.
    coords_center: array-like
        Coordinates of the regions to be patched (centroids of cell regions).
    target_patch_size: int
        Size of the patches to extract from the image.
    barcodes: array-like
        Barcodes associated with patches.
    show_extracted_images: bool, optional (default=False)
        If True, will show extracted images during the visualization phase.
    vis_width: int, optional (default=1000)
        Width of the visualization images.
    """
    # Path for the .h5 image dataset
    import os
    h5_path = os.path.join(patch_save_dir, name_data + '.h5')
    if os.path.exists(h5_path):
        print(h5_path, "exists")
        return
    else:
        print(h5_path, "does not exist")
        
    # Load the image and transpose it to the correct format
    print("Loading imgs ...")
    import numpy as np
    if 'tif_HE' in sdata:
        intensity_image = sdata['tif_HE'].copy()
    else:
        intensity_image = np.transpose(sdata['HE_original'].to_numpy(), (1, 2, 0))
    
    print("Patching: create image dataset (X) ...")
    # Create the patcher object to extract patches (localized square sub-region of an image) from an image at specified coordinates.
    from Patcher import Patcher
    patcher = Patcher(
        image=intensity_image,
        coords=coords_center,
        patch_size_target=target_patch_size
    )

    # Build and Save patches to an HDF5 file
    patcher.to_h5(h5_path, extra_assets={'barcode': barcodes})

    # Visualization
    print("Visualization")
    if show_extracted_images:
        print("Extracted Images (high time and memory consumption...)")
        path1 = os.path.join(patch_save_dir, name_data + '_patch_locations_on_image.png')
        patcher.save_visualization(path1, vis_width=vis_width)

        print("Spatial coordinates")
        path2 = os.path.join(patch_save_dir, name_data + '_spatial_coordinates.png')
        patcher.view_coord_points(path2, vis_width=vis_width)
    
        # Display some example images from the created dataset
        print("Examples from the created .h5 dataset")
        assets, _ = read_assets_from_h5(h5_path)
    
        n_images = 3
        fig, axes = plt.subplots(1, n_images, figsize=(15, 5))
        for i in range(n_images):
            axes[i].imshow(assets["img"][i])
        for ax in axes:
            ax.axis('off')
        path3 = os.path.join(patch_save_dir, name_data + '_patch_examples.png')
        plt.savefig(path3, dpi=dpi, bbox_inches='tight')
        plt.show()

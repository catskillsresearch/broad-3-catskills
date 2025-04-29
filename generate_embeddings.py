"""
**`generate_embeddings`**: A utility for generating embeddings from images  and saving them to an HDF5 file. It handles creating a `DataLoader`, running the model in evaluation mode and saving embeddings to disk.
"""

def generate_embeddings(embed_path, encoder, device, tile_h5_path, batch_size, num_workers):
    """
    Generate embeddings for images and save to a specified path.

    Parameters:
    -----------
    embed_path : str
        Path to save the embeddings.
    encoder : torch.nn.Module
        The encoder model for generating embeddings.
    device : torch.device
        Device to use for computation (e.g., 'cuda' or 'cpu').
    tile_h5_path : str
        Path to the HDF5 file containing images.
    batch_size : int
        Batch size for the DataLoader.
    num_workers : int
        Number of worker threads for data loading.
    overwrite : bool, optional
        If True, overwrite existing embeddings. Default is False.
    """

    # If the embeddings file doesn't exist or overwrite is True, proceed to generate embeddings
    import os

    # Set encoder to evaluation mode and move it to the device
    encoder.eval()
    encoder.to(device)

    # Create dataset and dataloader for tiles
    from H5Dataset import H5Dataset
    tile_dataset = H5Dataset(tile_h5_path, chunk_size=batch_size, img_transform=encoder.eval_transforms)

    import torch
    tile_dataloader = torch.utils.data.DataLoader(
        tile_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )

    # Generate and save embeddings
    from embed_tiles import embed_tiles
    return embed_tiles(tile_dataloader, encoder, embed_path, device, encoder.precision)

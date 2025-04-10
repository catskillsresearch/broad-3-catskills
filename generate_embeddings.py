"""
**`generate_embeddings`**: A utility for generating embeddings from images  and saving them to an HDF5 file. It handles creating a `DataLoader`, running the model in evaluation mode and saving embeddings to disk.
"""

import torch, os
from tqdm import tqdm
from utilities import *
import numpy as np
from H5Dataset import *
from save_hdf5 import *

def embed_tiles(dataloader, model: torch.nn.Module, embedding_save_path: str, device: str, precision):
    """
    Extracts embeddings from image tiles using the specified model and saves them to an HDF5 file.

    Parameters:
    -----------
    dataloader : torch.utils.data.DataLoader
        DataLoader providing the batches of image tiles.
    model : torch.nn.Module
        The model used to generate embeddings from the tiles.
    embedding_save_path : str
        Path where the generated embeddings will be saved.
    device : str
        The device to run the model on (e.g., 'cuda' or 'cpu').
    precision : torch.dtype
        The precision (data type) to use for inference (e.g., float16 for mixed precision).
    """

    model.eval()
    # Iterate over the batches in the DataLoader
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = post_collate_fn(batch)
        imgs = batch['imgs'].to(device).float()
        # Apply model on images
        with torch.inference_mode():
            if torch.cuda.is_available():  # Use mixed precision only if CUDA is available
                with torch.amp.autocast('cuda', dtype=precision):
                    embeddings = model(imgs)
            else:  # No mixed precision on CPU
                embeddings = model(imgs)

        # Set mode to 'w' for the first batch, 'a' for appending subsequent batches
        mode = 'w' if batch_idx == 0 else 'a'

        # Create a dictionary with embeddings and other relevant data to save
        asset_dict = {'embeddings': embeddings.cpu().numpy()}
        asset_dict.update({key: np.array(val) for key, val in batch.items() if key != 'imgs'})

        # Save the embeddings to the HDF5 file
        save_hdf5(embedding_save_path, asset_dict=asset_dict, mode=mode)

    return embedding_save_path


def generate_embeddings(embed_path, encoder, device, tile_h5_path, batch_size, num_workers, overwrite=False):
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
    if not os.path.isfile(embed_path) or overwrite:
        print(f"Generating embeddings for {embed_path} ...")

        # Set encoder to evaluation mode and move it to the device
        encoder.eval()
        encoder.to(device)

        # Create dataset and dataloader for tiles
        tile_dataset = H5Dataset(tile_h5_path, chunk_size=batch_size, img_transform=encoder.eval_transforms)
        tile_dataloader = torch.utils.data.DataLoader(
            tile_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers
        )

        # Generate and save embeddings
        embed_tiles(tile_dataloader, encoder, embed_path, device, encoder.precision)
    else:
        print(f"Skipping embedding {os.path.basename(embed_path)} as it already exists")
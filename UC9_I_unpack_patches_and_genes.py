import os, luigi
import spatialdata as sd  # Manage multi-modal spatial omics datasets
import numpy as np
from UC9_I_HistologyArchive import UC9_I_HistologyArchive
from extract_spatial_positions import extract_spatial_positions
from read_assets_from_h5 import read_assets_from_h5
import matplotlib.pylab as plt

class UC9_I_unpack_patches_and_genes(luigi.Task):
    def requires(self):
        return UC9_I_HistologyArchive()
        
    def output(self):
        return {
            'patches': luigi.LocalTarget('resources/run/UC9_I_patches.h5'),
            'genes': luigi.LocalTarget('resources/run/UC9_I_genes.h5ad'),
            'barcodes': luigi.LocalTarget('resources/run/UC9_I_barcodes.npy'),
            'spatial_positions': luigi.LocalTarget('resources/run/spatial_positions.npy'),
            'patch_locations_on_image': luigi.LocalTarget('resources/run/UC9_I_patch_locations_on_image.png'),
            'spatial_coordinates': luigi.LocalTarget('resources/run/UC9_I_spatial_coordinates.png'),
            'patch_examples': luigi.LocalTarget('resources/run/UC9_I_patch_examples.png')        }

    def spatial_positions(self):
        self.sdata = sd.read_zarr(self.input().path)   
        cell_id_train = self.sdata['anucleus'].obs["cell_id"].values
        new_spatial_coord = extract_spatial_positions(self.sdata, cell_id_train)
        self.sdata['anucleus'].obsm['spatial'] = new_spatial_coord
        np.save(self.output()['spatial_positions'].path, new_spatial_coord)

    def gene_expressions(self):
        # Create the gene expression dataset (Y)
        rows_to_keep = list(self.sdata['anucleus'].obs.sample(n=len(self.sdata['anucleus'].obs)).index)
        self.y_subtracted = self.sdata['anucleus'][rows_to_keep].copy()
        # Trick to set all index to same length to avoid problems when saving to h5
        self.y_subtracted.obs.index = ['x' + str(i).zfill(6) for i in self.y_subtracted.obs.index]
        # Check
        for index in self.y_subtracted.obs.index:
            if len(index) != len(self.y_subtracted.obs.index[0]):
                warnings.warn("indices of self.y_subtracted.obs should all have the same length to avoid problems when saving to h5", UserWarning)
        # Save the gene expression data to an H5AD file
        self.y_subtracted.write(self.output()['genes'].path)

    def patches(self):
        # Extract spatial coordinates and barcodes (cell IDs) for the patches
        coords_center = self.y_subtracted.obsm['spatial']
        barcodes = np.array(self.y_subtracted.obs.index)
        # Load the image and transpose it to the correct format
        intensity_image = np.transpose(self.sdata['HE_original'].to_numpy(), (1, 2, 0))
        from Patcher import Patcher
        self.patcher = Patcher(image=intensity_image, coords=coords_center, patch_size_target=32)
        self.patcher.to_h5(self.output()['patches'].path, extra_assets={'barcode': barcodes})
        np.save(self.output()['barcodes'].path, barcodes)
        
    def visualization(self):
        # Visualization
        vis_width=1000
        self.patcher.save_visualization(self.output()['patch_locations_on_image'].path, vis_width=vis_width)
        self.patcher.view_coord_points(self.output()['spatial_coordinates'].path, vis_width=vis_width)
        # Display some example images from the created dataset
        print("Examples from the created .h5 dataset")
        assets, _ = read_assets_from_h5(self.output()['patches'].path)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            axes[i].imshow(assets["img"][i])
        for ax in axes:
            ax.axis('off')
        plt.savefig(self.output()['patch_examples'].path, dpi=150, bbox_inches='tight')
        
    def run(self):
        self.spatial_positions()
        self.gene_expressions()
        self.patches()
        self.visualization()

if __name__ == "__main__":
    luigi.build(
        [UC9_I_unpack_patches_and_genes()],  # Replace with your task class
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

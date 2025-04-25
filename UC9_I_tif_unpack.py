import os, luigi
import spatialdata as sd  # Manage multi-modal spatial omics datasets
import numpy as np
from extract_spatial_positions import extract_spatial_positions
from read_assets_from_h5 import read_assets_from_h5
import matplotlib.pylab as plt
from skimage.measure import regionprops
from tqdm import tqdm
from Patcher import Patcher

class UC9_I_unpack_patches_and_genes(luigi.Task):
    
    def requires(self):
        from UC9_I_tif import UC9_I_tif
        return UC9_I_tif()
        
    def output(self):
        return {
            'UC9_I_dysplasia': luigi.LocalTarget('resources/run/UC9_I_cancer.h5'),
            'UC9_I_non_dysplasia': luigi.LocalTarget('resources/run/UC9_I_no_cancer.h5') }
        
    def get_images_and_regions(self):
        import skimage.io
        # Read the dysplasia-related images and store them in a dictionary
        self.dysplasia_img_list = {}
        for key in self.input():
            self.dysplasia_img_list[key] = skimage.io.imread(self.input()[key].path)
        self.regions = regionprops(self.dysplasia_img_list['tif_HE_nuc'])

    def get_cell_ids_by_type(self):
        # Divide cell IDs between dysplasia and non-dysplasia status
        self.cell_ids_no_cancer, self.cell_ids_cancer = [], []
        # Loop through each region and extract centroid if the cell ID matches
        tif_region = self.dysplasia_img_list['tif_region']
        for props in tqdm(self.regions):
            cell_id = props.label
            centroid = props.centroid
            y_center, x_center = int(centroid[0]), int(centroid[1])
            dysplasia = tif_region[y_center, x_center]
            if dysplasia == 1:
                self.cell_ids_no_cancer.append(cell_id)
            elif dysplasia == 2:
                self.cell_ids_cancer.append(cell_id)

    def save_patches(self, name_data, cell_ids):
        target_patch_size=32
        vis_width=1000
        h5_path = self.output()[name_data].path
        coords_center = extract_spatial_positions(self.dysplasia_img_list, cell_ids)
        barcodes = np.array(['x' + str(i).zfill(6) for i in list(cell_ids)]) 
        intensity_image = self.dysplasia_img_list['tif_HE'].copy()
        patcher = Patcher(
            image=intensity_image,
            coords=coords_center,
            patch_size_target=target_patch_size)
        patcher.to_h5(h5_path, extra_assets={'barcode': barcodes})

    def run(self):
        self.get_images_and_regions()
        self.get_cell_ids_by_type()
        self.save_patches("UC9_I_non_dysplasia", self.cell_ids_no_cancer)
        self.save_patches("UC9_I_dysplasia", self.cell_ids_cancer)

if __name__ == "__main__":
    luigi.build(
        [UC9_I_unpack_patches_and_genes()],  # Replace with your task class
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

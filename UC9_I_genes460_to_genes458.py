import os, luigi, json
from UC9_I_unpack_patches_and_genes import UC9_I_unpack_patches_and_genes
from load_adata import load_adata
import numpy as np

class UC9_I_genes460_to_genes458(luigi.Task):
    def requires(self):
        return UC9_I_unpack_patches_and_genes()
        
    def output(self):
        return luigi.LocalTarget('resources/run/UC9_I_genes458.npz')

    def run(self):
        with open('resources/run/genes458.json', 'r') as f:
            genes458 = json.load(f)
        barcodes = np.load(self.input()['barcodes'].path,allow_pickle=True)
        adata = load_adata(self.input()['genes'].path, genes=genes458, barcodes=barcodes, normalize=False)
        np.savez_compressed(self.output().path, adata.values)

if __name__ == "__main__":
    luigi.build(
        [UC9_I_genes460_to_genes458()],  # Replace with your task class
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

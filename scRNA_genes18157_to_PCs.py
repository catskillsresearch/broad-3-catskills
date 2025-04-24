import os, luigi
from template_pca_fit_transform import template_pca_fit_transform
from SCRNA_unpack import SCRNA_unpack

class scRNA_genes18157_to_PCs(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return template_pca_fit_transform(
            object_type="genes18157",
            object_name="scRNA",
            mse_goal=0.064,
            dependency_task=SCRNA_unpack,
            sample_size = 200,
            sub_input="scRNA_18157_gene_expressions")

    def run(self):
        pass

    def output(self):
        return self.requires().output()

if __name__ == "__main__":
    luigi.build(
        [scRNA_genes18157_to_PCs()], 
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

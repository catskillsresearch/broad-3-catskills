import os, luigi
from template_pca_transform import template_pca_transform
from UC9_I_genes458_to_PCs import UC9_I_genes458_to_PCs
from UC9_I_genes460_to_genes458 import UC9_I_genes460_to_genes458
from SCRNA_unpack import SCRNA_unpack

class scRNA_Genes_458_to_UC9_I_Genes_458_basis_PCs(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return template_pca_transform(
            object_type = "genes458",
            object_name = "scRNA",
            pca_fit_transform = UC9_I_genes458_to_PCs,
            source = SCRNA_unpack,
            source_field = 'scRNA_458_gene_expressions')
        
    def run(self):
        pass

    def output(self):
        return self.requires().output()

if __name__ == "__main__":
    luigi.build(
        [scRNA_Genes_458_to_UC9_I_Genes_458_basis_PCs()], 
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

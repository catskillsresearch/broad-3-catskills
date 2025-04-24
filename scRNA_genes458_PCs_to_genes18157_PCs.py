import os, luigi
from template_ridge_fit import template_ridge_fit
from scRNA_Genes_458_to_UC9_I_Genes_458_basis_PCs import scRNA_Genes_458_to_UC9_I_Genes_458_basis_PCs
from scRNA_genes18157_to_PCs import scRNA_genes18157_to_PCs

class scRNA_genes458_PCs_to_genes18157_PCs(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return template_ridge_fit(
            src_object_type = "genes458_PCs",
            src_object_name = "scRNA",
            src_task = scRNA_Genes_458_to_UC9_I_Genes_458_basis_PCs,
            tgt_object_type = "genes18157_PCs",
            tgt_object_name = "scRNA", 
            tgt_task = scRNA_genes18157_to_PCs)

    def run(self):
        pass

    def output(self):
        return self.requires().output()
    
if __name__ == "__main__":
    luigi.build(
        [scRNA_genes458_PCs_to_genes18157_PCs()],
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

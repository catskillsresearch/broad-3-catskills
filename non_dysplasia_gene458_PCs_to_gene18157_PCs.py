import os, luigi
from template_ridge_transform import template_ridge_transform
from non_dysplasia_feature_PCs_to_gene_PCs import non_dysplasia_feature_PCs_to_gene_PCs
from scRNA_genes458_PCs_to_genes18157_PCs import scRNA_genes458_PCs_to_genes18157_PCs

class non_dysplasia_gene458_PCs_to_gene18157_PCs(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return template_ridge_transform(
            src_task = non_dysplasia_feature_PCs_to_gene_PCs,
            fit_task = scRNA_genes458_PCs_to_genes18157_PCs,
            tgt_object_type = "genes18157_PCs",
            tgt_object_name = "UC9_I_non_dysplasia")

    def run(self):
        pass

    def output(self):
        return self.requires().output()
    
if __name__ == "__main__":
    luigi.build(
        [non_dysplasia_gene458_PCs_to_gene18157_PCs()],
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

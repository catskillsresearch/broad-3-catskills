import os, luigi
from template_pca_inverse_transform import template_pca_inverse_transform
from scRNA_genes18157_to_PCs import scRNA_genes18157_to_PCs
from dysplasia_gene458_PCs_to_gene18157_PCs import dysplasia_gene458_PCs_to_gene18157_PCs

class dysplasia_genes18157PCs_to_genes18157(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return template_pca_inverse_transform(
            object_type="genes18157",
            object_name="UC9_I_dysplasia",
            pca_fit_transform = scRNA_genes18157_to_PCs,
            source = dysplasia_gene458_PCs_to_gene18157_PCs,
            source_field = 'UC9_I_dysplasia')

    def run(self):
        pass

    def output(self):
        return self.requires().output()

if __name__ == "__main__":
    luigi.build(
        [dysplasia_genes18157PCs_to_genes18157()],  
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

import os, luigi
from template_pca_inverse_transform import template_pca_inverse_transform
from UC9_I_genes458_to_PCs import UC9_I_genes458_to_PCs
from dysplasia_feature_PCs_to_gene_PCs import dysplasia_feature_PCs_to_gene_PCs

class dysplasia_genes458PCs_to_genes458(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return template_pca_inverse_transform(
            object_type="genes458",
            object_name="UC9_I_dysplasia",
            pca_fit_transform = UC9_I_genes458_to_PCs,
            source = dysplasia_feature_PCs_to_gene_PCs,
            source_field = 'UC9_I_dysplasia')

    def run(self):
        pass

    def output(self):
        return self.requires().output()

if __name__ == "__main__":
    luigi.build(
        [dysplasia_genes458PCs_to_genes458()],  
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

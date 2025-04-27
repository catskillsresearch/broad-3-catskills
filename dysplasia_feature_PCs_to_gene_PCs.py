import os, luigi
from template_ridge_transform import template_ridge_transform
from dysplasia_features_to_PCs import dysplasia_features_to_PCs
from UC9_I_feature_PCs_to_gene_PCs import UC9_I_feature_PCs_to_gene_PCs

class dysplasia_feature_PCs_to_gene_PCs(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return template_ridge_transform(
            src_task = dysplasia_features_to_PCs,
            fit_task = UC9_I_feature_PCs_to_gene_PCs,
            tgt_object_type = "genes458_PCs",
            tgt_object_name = "UC9_I_dysplasia")

    def run(self):
        pass

    def output(self):
        return self.requires().output()
    
if __name__ == "__main__":
    luigi.build(
        [dysplasia_feature_PCs_to_gene_PCs()],
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

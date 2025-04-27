import os, luigi
from template_pca_transform import template_pca_transform
from UC9_I_features_to_PCs import UC9_I_features_to_PCs
from dysplasia_patches_to_features import dysplasia_patches_to_features

class dysplasia_features_to_PCs(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return template_pca_transform(
            object_type="feature",
            object_name="UC9_I_dysplasia",
            pca_fit_transform = UC9_I_features_to_PCs,
            source = dysplasia_patches_to_features,
            source_field = 'UC9_I_dysplasia')

    def run(self):
        pass

    def output(self):
        return self.requires().output()

if __name__ == "__main__":
    luigi.build(
        [dysplasia_features_to_PCs()],  
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

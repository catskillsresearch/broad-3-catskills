import os, luigi
from template_pca_fit_transform import template_pca_fit_transform
from UC9_I_patches_to_features import UC9_I_patches_to_features

class UC9_I_features_to_PCs(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return template_pca_fit_transform(
            object_type="feature",
            object_name="UC9_I",
            mse_goal=0.16,
            dependency_task=UC9_I_patches_to_features,
            sub_input = None,
            sample_size = 1000
        )

    def run(self):
        pass

    def output(self):
        return self.requires().output()

if __name__ == "__main__":
    luigi.build(
        [UC9_I_features_to_PC()],  
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

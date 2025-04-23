import os, luigi
from UC9_I_object_to_PCs import UC9_I_object_to_PCs
from UC9_I_patches_to_features import UC9_I_patches_to_features

class UC9_I_features_to_PCs(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return UC9_I_object_to_PCs(
            object_type="feature",
            mse_goal=0.16,
            dependency_task=UC9_I_patches_to_features  # Must be a Task class
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

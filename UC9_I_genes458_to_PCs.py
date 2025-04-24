import os, luigi
from template_pca_fit_transform import template_pca_fit_transform
from UC9_I_genes460_to_genes458 import UC9_I_genes460_to_genes458

class UC9_I_genes458_to_PCs(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return template_pca_fit_transform(
            object_type="genes458",
            object_name="UC9_I",
            mse_goal=0.064,
            dependency_task=UC9_I_genes460_to_genes458,
            sub_input = None

    def run(self):
        pass

    def output(self):
        return self.requires().output()

if __name__ == "__main__":
    luigi.build(
        [UC9_I_genes458_to_PCs()], 
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

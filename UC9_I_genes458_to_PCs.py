import os, luigi
from UC9_I_object_to_PCs import UC9_I_object_to_PCs
from UC9_I_genes460_to_genes458 import UC9_I_genes460_to_genes458

class UC9_I_genes458_to_PCs(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return UC9_I_object_to_PCs(
            object_type="genes458",
            mse_goal=0.064,
            dependency_task=UC9_I_genes460_to_genes458)

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

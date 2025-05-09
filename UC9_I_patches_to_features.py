import luigi
from UC9_I_unpack_patches_and_genes import UC9_I_unpack_patches_and_genes
from template_patches_to_features import template_patches_to_features

class UC9_I_patches_to_features(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return template_patches_to_features(patches_task = UC9_I_unpack_patches_and_genes, patch_field = 'patches', name = 'UC9_I')

    def run(self):
        pass

    def output(self):
        return self.requires().output()
    
if __name__ == "__main__":
    luigi.build(
        [UC9_I_patches_to_features()], 
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

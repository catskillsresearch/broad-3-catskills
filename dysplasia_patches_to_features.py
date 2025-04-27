import luigi
from UC9_I_tif_unpack import UC9_I_tif_unpack
from template_patches_to_features import template_patches_to_features

class dysplasia_patches_to_features(luigi.Task):
    def requires(self):
        # Provide hardcoded parameters and dependency
        return template_patches_to_features(patches_task = UC9_I_tif_unpack, patch_field = 'UC9_I_dysplasia', name = 'UC9_I_dysplasia')

    def run(self):
        pass

    def output(self):
        return self.requires().output()

if __name__ == "__main__":
    luigi.build(
        [dysplasia_patches_to_features()],
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

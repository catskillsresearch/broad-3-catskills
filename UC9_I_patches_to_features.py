import luigi
from UC9_I_unpack_patches_and_genes import UC9_I_unpack_patches_and_genes
from template_patches_to_features import template_patches_to_features

if __name__ == "__main__":
    luigi.build(
        [template_patches_to_features(patches_task = UC9_I_unpack_patches_and_genes)], 
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

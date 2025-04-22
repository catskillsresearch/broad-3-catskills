import os, luigi
from UC9_I_object_to_PCs import UC9_I_object_to_PCs
from UC9_I_patches_to_features import UC9_I_patches_to_features

if __name__ == "__main__":
    luigi.build(
        [UC9_I_object_to_PCs("feature", 0.16, UC9_I_patches_to_features)],  
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

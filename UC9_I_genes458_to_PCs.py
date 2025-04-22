import os, luigi
from UC9_I_object_to_PCs import UC9_I_object_to_PCs
from UC9_I_genes460_to_genes458 import UC9_I_genes460_to_genes458

if __name__ == "__main__":
    luigi.build(
        [UC9_I_object_to_PCs("genes458", UC9_I_genes460_to_genes458)],  
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

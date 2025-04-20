import os, json
import spatialdata as sd 
import scanpy as sc
import numpy as np
from log1p_normalization_scale_factor import log1p_normalization_scale_factor
import luigi

class UC9_I_HistologyArchive(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget("data/UC9_I.zarr")  # UC9_I histology image

class scRNA_SequenceArchive(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('data/Crunch3_scRNAseq.h5ad')  # UC9_I histology image


class SCRNA_unpack(luigi.Task):
    def requires(self):
        return {
            'UC9_I': UC9_I_HistologyArchive(),  # Must expose outputs via output() method
            'scRNA': scRNA_SequenceArchive()
        }
        
    def output(self):
        return {
            'genes460': luigi.LocalTarget('resources/run/genes460.json'),
            'genes458': luigi.LocalTarget('resources/run/genes458.json'),
            'genes18615': luigi.LocalTarget('resources/run/genes18615.json'),
            'scRNA_458_gene_expressions': luigi.LocalTarget('resources/run/scRNA_458_gene_expressions.npz'),
            'scRNA_18157_gene_expressions': luigi.LocalTarget('resources/run/scRNA_18157_gene_expressions.npz'),
        }
    
    def run(self):
        rundir = 'resources/run'
        os.makedirs(rundir, exist_ok=True)
        
        sdata = sd.read_zarr(self.input()['UC9_I'].path)
        genes460 = sdata['anucleus'].var['gene_symbols'].values.tolist()

        scRNAseq = sc.read_h5ad(self.input()['scRNA'].path)
        genes18615 = list(scRNAseq.var.index)
        genes458 = [x for x in genes460 if x in genes18615]
        genes18157 = [gene for gene in genes18615 if gene not in genes458]

        # scRNA-Seq data log1p-normalized with scale factor 10000 on 18615 genes
        rna_data_norm_10000_unmeasured_genes = scRNAseq[:, genes18157].X.toarray()
        
        # scRNA-Seq data log1p-normalized with scale factor 100 on 460 genes
        rna_data_norm_100_common_genes = log1p_normalization_scale_factor(
                scRNAseq[:, genes458].layers["counts"].toarray(), 
                scale_factor=100)
        with self.output()['genes460'].open('w') as f:
            json.dump(genes460, f, indent=4)
        with self.output()['genes458'].open('w') as f:
            json.dump(genes458, f, indent=4)
        with self.output()['genes18615'].open('w') as f:
            json.dump(genes18615, f, indent=4)
        with self.output()['genes18157'].open('w') as f:
            json.dump(genes18157, f, indent=4)

        np.savez_compressed(fself.output()['scRNA_18157_gene_expressions'].path, 
                            my_array=rna_data_norm_10000_unmeasured_genes)

        np.savez_compressed(f'{rundir}/scRNA_458_gene_expressions', 
                            my_array=rna_data_norm_100_common_genes)

if __name__ == "__main__":
    luigi.build(
        [SCRNA_unpack()],  # Replace with your task class
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

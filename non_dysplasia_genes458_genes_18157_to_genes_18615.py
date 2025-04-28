import luigi
from dysplasia_genes458PCs_to_genes458 import dysplasia_genes458PCs_to_genes458
from dysplasia_genes18157PCs_to_genes18157 import dysplasia_genes18157PCs_to_genes18157

class dysplasia_genes458_genes_18157_to_genes_18615(luigi.Task):
    def requires(self):
        return {'genes458': dysplasia_genes458PCs_to_genes458(),
                'genes18157': dysplasia_genes18157PCs_to_genes18157()}

    def output(self):
        return luigi.LocalTarget(f'resources/run/UC9_I_dysplasia_genes18615.npz'),
        
    def run(self):
        d458fn = self.input()['genes458'].path
        d157fn = self.input()['genes18157'].path
        d458 = np_loadz(d458fn)
        d157 = np_loadz(d157fn)
        d18615 = np.hstack([d458, d157])
        np.savez_compressed(self.output().path, d18615)

if __name__ == "__main__":
    luigi.build(
        [dysplasia_genes458_genes_18157_to_genes_18615()],
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

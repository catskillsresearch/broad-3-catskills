import luigi

class UC9_I_HistologyArchive(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget("data/UC9_I.zarr")  # UC9_I histology image

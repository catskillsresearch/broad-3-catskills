import luigi

class UC9_I_tif(luigi.ExternalTask):
    
    def output(self):
        return {'tif_HE': luigi.LocalTarget("data/UC9_I-crunch3-HE.tif"),
                'tif_HE_nuc': luigi.LocalTarget("data/UC9_I-crunch3-HE-label-stardist.tif"),
                'tif_region': luigi.LocalTarget("data/UC9_I-crunch3-HE-dysplasia-ROI.tif")}

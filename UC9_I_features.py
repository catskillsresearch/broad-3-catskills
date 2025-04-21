import luigi

class UC9_I_features(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('resources/run/UC9_I_features.h5')
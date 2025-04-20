import luigi

class Resnet50ModelFile(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('resources/pytorch_model.bin')
import luigi

class dysplasia_patches_to_features(luigi.Task):
    def requires(self):
        return {'genes658': 
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

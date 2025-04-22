import os, luigi, torch
from inf_encoder_factory import inf_encoder_factory
from generate_embeddings import generate_embeddings
from UC9_I_unpack_patches_and_genes import UC9_I_unpack_patches_and_genes
from Resnet50ModelFile import Resnet50ModelFile

class UC9_I_patches_to_features(luigi.Task):
    def requires(self):
        return {
                'weights': Resnet50ModelFile(),
                'patches': UC9_I_unpack_patches_and_genes() }
        
    def output(self):
        return luigi.LocalTarget('resources/run/UC9_I_features.csv.gz')

    def run(self):
        encoder = inf_encoder_factory("resnet50")(self.input()['weights'].path)
        patches_path = self.input()['patches']['patches'].path
        embed_path = self.output().path
        batch_size = 128
        num_workers = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generate_embeddings(embed_path, encoder, device, patches_path, batch_size, num_workers)

if __name__ == "__main__":
    luigi.build(
        [UC9_I_patches_to_features()],  # Replace with your task class
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

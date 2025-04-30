import os, luigi, torch
from inf_encoder_factory import inf_encoder_factory
from generate_embeddings import generate_embeddings
from Resnet50ModelFile import Resnet50ModelFile
from select_random_from_2D_array import select_random_from_2D_array
import matplotlib.pyplot as plt
import numpy as np

class template_patches_to_features(luigi.Task):

    patches_task = luigi.TaskParameter()
    patch_field = luigi.Parameter()
    name = luigi.Parameter()
    
    def requires(self):
        return {'weights': Resnet50ModelFile(),
                'patches': self.patches_task() }
        
    def output(self):
        tag = f'{self.name}_features'
        return luigi.LocalTarget(f'resources/run/{tag}.npz')

    def run(self):
        encoder = inf_encoder_factory("resnet50")(self.input()['weights'].path)
        patches_path = self.input()['patches'][self.patch_field].path
        embed_path = self.output()['features'].path
        batch_size = 128
        num_workers = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        generate_embeddings(embed_path, encoder, device, patches_path, batch_size, num_workers)

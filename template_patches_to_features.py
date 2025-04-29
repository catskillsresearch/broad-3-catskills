import os, luigi, torch
from inf_encoder_factory import inf_encoder_factory
from generate_embeddings import generate_embeddings
from Resnet50ModelFile import Resnet50ModelFile

class template_patches_to_features(luigi.Task):

    patches_task = luigi.TaskParameter()
    patch_field = luigi.Parameter()
    name = luigi.Parameter()
    
    def requires(self):
        return {
                'weights': Resnet50ModelFile(),
                'patches': self.patches_task() }
        
    def output(self):
        return {'features': luigi.LocalTarget(f'resources/run/{self.name}_features.npz'),
                'density': luigi.LocalTarget(f'mermaid/{self.name}_features_density.png')}

    def show_density(self, data):
        data_flat = select_random_from_2D_array(data, 1000)
        plt.hist(data_flat, bins=100, density=True)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(f"Non-0 {self.name} density")
        out = self.output()
        plt.savefig(out['density'].path, dpi=150, bbox_inches='tight')
        plt.clf()

    def run(self):
        encoder = inf_encoder_factory("resnet50")(self.input()['weights'].path)
        patches_path = self.input()['patches'][self.patch_field].path
        embed_path = self.output().path
        batch_size = 128
        num_workers = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data = generate_embeddings(embed_path, encoder, device, patches_path, batch_size, num_workers)
        self.show_density(data)
import luigi, os, json
from glob import glob
from np_loadz import np_loadz
import numpy as np
import pandas as pd
from gene_ranking import gene_ranking
import matplotlib.pyplot as plt
from dysplasia_genes458_genes_18157_to_genes_18615 import dysplasia_genes458_genes_18157_to_genes_18615
from non_dysplasia_genes458_genes_18157_to_genes_18615 import non_dysplasia_genes458_genes_18157_to_genes_18615
from SCRNA_unpack import SCRNA_unpack

class genes_ranked_by_descending_abs_log_fold_change(luigi.Task):
    def requires(self):
        return {'dysplasia': dysplasia_genes458_genes_18157_to_genes_18615(),
                'non_dysplasia': non_dysplasia_genes458_genes_18157_to_genes_18615(),
                'unpack': SCRNA_unpack()}

    def output(self):
        return {'prediction': luigi.LocalTarget('resources/prediction.csv'),
                'gene_ranking': luigi.LocalTarget('resources/gene_ranking.csv'),
                'logFC_plot': luigi.LocalTarget('resources/logFC_plot.csv')}

    def logFC_plot(self, genes460, prediction, df_gene_ranking):
        prediction['is460'] = prediction['Gene Name'].apply(lambda x: x in genes460)
        df = df_gene_ranking[['logFC']].copy()
        df['is460'] = ['green' if gene in genes460 else 'red' for gene in df.index]
        df['rank'] = [i+1 for i in range(len(df))]

        plt.figure(figsize=(10, 6))
        df_460 = df[df.is460 == 'green']
        df_imputed = df[df.is460 != 'green']
        plt.scatter(df_imputed['rank'].values, df_imputed['logFC'].values, c=df_imputed['is460'].values, s=0.1);
        plt.scatter(df_460['rank'].values, df_460['logFC'].values, c=df_460['is460'].values, s=5);
        
        # Add labels and title
        plt.xlabel('Sort index by descending abs log fold change')
        plt.ylabel('logFC')
        plt.title('Scatterplot of logFC vs Imputed or Assayed')
        
        # Create custom legend
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='red', label='Imputed')
        green_patch = mpatches.Patch(color='green', label='Assayed')
        plt.legend(handles=[red_patch, green_patch]);
        
        plt.savefig('resources/logFC_plot.png', dpi=300, bbox_inches='tight')  

    def rank_genes(self):
        inp = self.input()
        d18615 = np_loadz(inp['dysplasia'].path)
        nd18615 = np_loadz(inp['non_dysplasia'].path)
        
        with open(inp['unpack']['genes18615'].path,'r') as f:
            genes = json.load(f)
        
        with open(inp['unpack']['genes460'].path, 'r') as f:
            genes460 = json.load(f)
        
        prediction_cell_ids_no_cancer = pd.DataFrame(nd18615, columns=genes)
        prediction_cell_ids_cancer = pd.DataFrame(d18615, columns=genes)
        prediction, df_gene_ranking = gene_ranking(prediction_cell_ids_no_cancer, prediction_cell_ids_cancer)
        
        # Save the ranked genes to a CSV file -> to use for the inder function and crunchDAO crunch 3 submission
        outp = self.output()
        prediction.to_csv(outp['gene_ranking'].path)
        df_gene_ranking.to_csv(outp['logFC_plot'].path)

        return genes460, prediction, df_gene_ranking 
        
    def run(self):
        genes460, prediction, df_gene_ranking = self.rank_genes()
        self.logFC_plot(genes460, prediction, df_gene_ranking)

if __name__ == "__main__":
    luigi.build(
        [genes_ranked_by_descending_abs_log_fold_change()],
        local_scheduler=True,  # Required for local execution
        workers=1  # Optional: single worker for serial execution
    )

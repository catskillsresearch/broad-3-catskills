# Broad Institute IBD Challenge
# Rank all 18615 protein-coding genes based on ability to distinguish dysplastic from non-cancerous tissue

import os

def train(
    data_directory_path: str,  # Path to the input data directory
    model_directory_path: str  # Path to save the trained model and results
):
    os.system('python genes_ranked_by_descending_abs_log_fold_change.py')

if __name__=="__main__":
    data_directory_path='./data'
    model_directory_path="./resources"
    train(data_directory_path, model_directory_path)

    


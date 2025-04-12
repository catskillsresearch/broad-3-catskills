import os
import pandas as pd

def infer(
    data_file_path: str,
    data_directory_path: str,
    model_directory_path: str
):
    return pd.read_csv(os.path.join(model_directory_path, "gene_ranking.csv"), index_col=0)

if __name__=="__main__":
    prediction = infer(
        data_file_path="./data",
        data_directory_path="./data",
        model_directory_path="./resources"
    )
    print(prediction.head())


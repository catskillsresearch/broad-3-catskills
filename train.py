from args1 import args1
from extract_spatial_positions import *
from gene_ranking import gene_ranking
from infer_crunch_1 import infer_crunch_1
from infer_crunch_2 import infer_crunch_2
from print_memory_usage import print_memory_usage
from run_training import *
from skimage.measure import regionprops
from tqdm import tqdm
from types import SimpleNamespace
import gc, os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc 
import skimage.io
import spatialdata as sd 

# Broad Institute IBD Challenge
# Rank all 18615 protein-coding genes based on ability to distinguish dysplastic from non-cancerous tissue

def train(
    data_directory_path: str,  # Path to the input data directory
    model_directory_path: str  # Path to save the trained model and results
):
   
    # 1. Prepare the input data by preprocessing the spatial transcriptomics datasets (image patches (X) and gene expression (Y)).
    args, dir_processed_dataset, dir_models_and_results, list_ST_name_data = args1(model_directory_path)
    preprocess_spatial_transcriptomics_data_train(list_ST_name_data, data_directory_path, dir_processed_dataset,
                                                  args.size_subset, args.target_patch_size, args.vis_width, args.show_extracted_images)
    # 2. Create leave-one-out cross-validation splits for training and testing.
    create_cross_validation_splits(dir_processed_dataset, n_fold=args.n_fold)

    # 3. Use the specified encoder model (ResNet50) and a regression method (ridge regression) to train the model
    # 4. Save the results and the trained model in the `.resources/` directories for later inference.
    #    All regression models and metric results are saved in `./resources/ST_pred_results`
    #    Preprocessed datasets are saved in `resources/processed_dataset`
    #    Official challenge metric: L2 mean error (`l2_errors_mean`)
    run_training(args)

    data_directory_path="./data"
    model_directory_path="./resources"
    args_dict = {
        "file_name_scRNAseq": 'Crunch3_scRNAseq.h5ad',  # Filename for scRNAseq data
        "filter_column_scRNAseq": "dysplasia",  # Column for filtering scRNAseq data
        "filter_value_no_cancer": "n",  # Filtered column value indicating absence of cancer
        "filter_value_cancer": "y",  # Filtered column value indicating the presence of cancer
        "column_for_ranking": "abs_logFC",
        "ascending": False
    }
    
    args = SimpleNamespace(**args_dict)
    sdata = sd.read_zarr(os.path.join(data_directory_path, 'UC9_I.zarr'))
    gene_460_names = list(sdata["anucleus"].var.index)
    
    dysplasia_file = {
        # H&E image of tissue with dysplasia
        'tif_HE': os.path.join(data_directory_path, 'UC9_I-crunch3-HE.tif'),
    
        # Nucleus segmentation of H&E image
        'tif_HE_nuc': os.path.join(data_directory_path, 'UC9_I-crunch3-HE-label-stardist.tif'),
    
        # Regions in H&E image highlighting dysplasia and non-dysplasia
        'tif_region': os.path.join(data_directory_path, 'UC9_I-crunch3-HE-dysplasia-ROI.tif')
    }
    

    # Read the dysplasia-related images and store them in a dictionary
    dysplasia_img_list = {}
    for key in dysplasia_file:
        dysplasia_img_list[key] = skimage.io.imread(dysplasia_file[key])
    

    regions = regionprops(dysplasia_img_list['tif_HE_nuc'])
    

    # Divide cell IDs between dysplasia and non-dysplasia status
    cell_ids_no_cancer, cell_ids_cancer = [], []
    # Loop through each region and extract centroid if the cell ID matches
    for props in tqdm(regions):
        cell_id = props.label
        centroid = props.centroid
        y_center, x_center = int(centroid[0]), int(centroid[1])
        # Using UC9_I-crunch3-HE-dysplasia-ROI.tif, check if cell ID highlight dysplasia or non-dysplasia (or 0 indicating other tissue regions)
        dysplasia = dysplasia_img_list['tif_region'][y_center, x_center]
        if dysplasia == 1:
            cell_ids_no_cancer.append(cell_id)
        elif dysplasia == 2:
            cell_ids_cancer.append(cell_id)
    
    prediction_cell_ids_no_cancer1 = infer_crunch_1(
        name_data="UC9_I no cancer",
        data_file_path=data_directory_path,
        model_directory_path=model_directory_path,
        sdata=dysplasia_img_list,
        cell_ids=cell_ids_no_cancer,
        gene_460_names=gene_460_names
    )
    
    prediction_cell_ids_cancer1 = infer_crunch_1(
        name_data="UC9_I cancer",
        data_file_path=data_directory_path,
        model_directory_path=model_directory_path,
        sdata=dysplasia_img_list,
        cell_ids=cell_ids_cancer,
        gene_460_names=gene_460_names
    )
    
    del regions
    del dysplasia_img_list
    

    scRNAseq = sc.read_h5ad(os.path.join(data_directory_path, args.file_name_scRNAseq))
    
    # Filter scRNAseq data by dysplasia status
    scRNAseq_no_cancer = scRNAseq[scRNAseq.obs[args.filter_column_scRNAseq] == args.filter_value_no_cancer].copy()
    scRNAseq_cancer = scRNAseq[scRNAseq.obs[args.filter_column_scRNAseq] == args.filter_value_cancer].copy()
    del scRNAseq
    gc.collect(); print_memory_usage("memory")
    
    subsample = 5000
    
    # No cancer status: predict the expression of 18157 genes using the expression of the 460 inferred genes and scRNAseq data (Crunch 2)
    # 
    # `Y` is log1p-normalized with scale factor 100.

    prediction_cell_ids_no_cancer2 = infer_crunch_2(
        prediction_460_genes=prediction_cell_ids_no_cancer1[0:subsample],
        name_data="UC9_I no cancer",
        data_file_path=data_directory_path,
        model_directory_path=model_directory_path,
        scRNAseq=scRNAseq_no_cancer,
        filter_column=args.filter_column_scRNAseq,
        filter_value=args.filter_value_no_cancer
    )
    
    gc.collect(); print_memory_usage("memory")
    
    # Cancer status: predict the expression of 18157 genes using the expression of the 460 inferred genes and scRNAseq data (Crunch 2)
    prediction_cell_ids_cancer2 = infer_crunch_2(
        prediction_460_genes=prediction_cell_ids_cancer1[0:subsample],
        name_data="UC9_I cancer",
        data_file_path=data_directory_path,
        model_directory_path=model_directory_path,
        scRNAseq=scRNAseq_cancer,
        filter_column=args.filter_column_scRNAseq,
        filter_value=args.filter_value_cancer
    )
    
    gc.collect(); print_memory_usage("memory")
    
    prediction, df_gene_ranking = gene_ranking(prediction_cell_ids_no_cancer2, prediction_cell_ids_cancer2,
                                   column_for_ranking=args.column_for_ranking, ascending=args.ascending)
    
    # Save the ranked genes to a CSV file -> to use for the inder function and crunchDAO crunch 3 submission
    prediction.to_csv(os.path.join(model_directory_path, "gene_ranking.csv"))
    
    prediction['is460'] = prediction['Gene Name'].apply(lambda x: x in gene_460_names)
    df = df_gene_ranking[['logFC']].copy()
    df['is460'] = ['green' if gene in gene_460_names else 'red' for gene in df.index]
    df['rank'] = [i+1 for i in range(len(df))]
    
    # Create the scatterplot
    plt.figure(figsize=(10, 6))
    df_460 = df[df.is460 == 'green']
    df_imputed = df[df.is460 != 'green']
    plt.scatter(df_imputed['rank'].values, df_imputed['logFC'].values, c=df_imputed['is460'].values, s=0.1);
    plt.scatter(df_460['rank'].values, df_460['logFC'].values, c=df_460['is460'].values, s=5);
    
    # Add labels and title
    plt.xlabel('Row Number')
    plt.ylabel('logFC')
    plt.title('Scatterplot of logFC vs Row Number')
    
    # Create custom legend
    red_patch = mpatches.Patch(color='red', label='Imputed')
    green_patch = mpatches.Patch(color='green', label='Assayed')
    plt.legend(handles=[red_patch, green_patch]);
    
    plt.savefig('resources/logFC_plot.png', dpi=300, bbox_inches='tight')
   
if __name__=="__main__":
    data_directory_path='./data'
    model_directory_path="./resources"
    train(data_directory_path, model_directory_path)

    


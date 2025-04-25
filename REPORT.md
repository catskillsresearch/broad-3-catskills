# Method Description

The method is as follows:

## Unpack scRNA data
![scRNA data unpacking](mermaid/SCRNA_unpack.png) 

## Unpack UC9_I patches and genes
![UC9_I_unpack_patches_and_genes](mermaid/UC9_I_unpack_patches_and_genes.png)  

## Template patches to features
![template_patches_to_features](mermaid/template_patches_to_features.png)  

## UC9_I patches to features 
Apply patches to features pipeline to UC9_I patches.

## Template principal component dimension reduction fit and transform pipeline
![template_pca_fit_transform](mermaid/template_pca_fit_transform.png)  

## UC9_I features to PCs
Apply principal component dimension reduction pipline to features for MSE <= 0.16.

## UC9_I Genes 460 to Genes 458
![UC9_I_genes460_to_genes458](mermaid/UC9_I_genes460_to_genes458.png)

## UC9_I Genes 458 to PCs
Apply principal component dimension reduction pipline to UCI_9 458 gene expressions for MSE <= 0.064.

## Template Ridge regression fit pipeline
![template_ridge_fit](mermaid/template_ridge_fit.png)

## UC9_I Regress Feature PCs to Gene PCs
Apply Ridge regression pipeline to create a linear map from Feature PCs to Gene PCS.

## Template PCA transform pipeline
![template_pca_transform](mermaid/template_pca_transform.png)

## scRNA Genes 458 to UC9_I Genes 458 basis PCs
Apply PCA transform pipeline with UC9_I Genes 458 PCA basis to map scRNA Genes 458 to PCs.

## scRNA Genes 18157 to PCs
Apply PCA fit transform pipeline to scRNA 18157 gene expressions for MSE <= 0.064.

## scRNA Regress Gene 458 PCs to Gene 18157 PCs
Apply Ridge regression pipeline to create a linear map from Gene 458 PCs to Gene 18157 PCS.

## UC9_I TIF Unpack
![UC9_I_tif_unpack](mermaid/UC9_I_tif_unpack.png)

## Dysplasia chips to features
Apply patches to features pipeline to Dysplasia chips.

## Non-dysplasia chips to features
Apply patches to features pipeline to Non-Dysplasia chips.

## scRNA Gene Predictor Transform (apply to plasia and non-dysplasia gene sets)
![scRNA gene predictor transform](mermaid/plasia_gene_inference.png)

## Genes ranked by highest absolute differential expression
![Crunch3 Processing](mermaid/crunch3.png)  

# Rationale

For the gene panel we suggest to take the top 500 genes in our sorted list which is ranked by absolute value of log fold change between dysplasia and non-dysplasia chips.

# Data and Resources Used

We use only the data provided by Broad Institute.  We do not use external data.  We use the registered images.  We don't do anything with the unregistered images.  We use the provided Crunch 3 dataset to select dysplasia and non-dysplasia cells for analysis.

We use the large dataset for UC9_I, the UC9_I tif files and the scRNA data.

We train on an Ubuntu PC with 1TB of SSD, an Intel Core i9 processor, 64GB of RAM, and an NVidia GTX 3060 with 12GB of VRAM.

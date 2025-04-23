# Method Description

The method is as follows:

## Unpack scRNA data
![scRNA data unpacking](mermaid/SCRNA_unpack.png) 

## Unpack UC9_I patches and genes
![UC9_I_unpack_patches_and_genes](mermaid/UC9_I_unpack_patches_and_genes.png)  

## UC9_I patches to features 
![UC9_I_patches_to_features](mermaid/UC9_I_patches_to_features.png)  

## Template principal component dimension reduction pipeline
![template_data_to_PCs](mermaid/template_data_to_PCs.png)  

## UC9_I features to PCs
Apply principal component dimension reduction pipline to features for MSE <= 0.16.

## UC9_I Genes 460 to Genes 458
![UC9_I_genes460_to_genes458](mermaid/UC9_I_genes460_to_genes458.png)

## UC9_I Genes 458 to PCs
Apply principal component dimension reduction pipline to 458 gene expressions for MSE <= 0.064.

## UC9_I Regress Feature PCs to Gene PCs
![UC9I gene predictor fit](mermaid/UC9_I_feature_PCs_to_gene_PCs.png)

## scRNA Gene Predictor Fit
![scRNA gene predictor fit](mermaid/SCRNA_calibration.png)

## UC9_I TIF splitter
![UC9_I TIF splitter](mermaid/UC9_I_tif_split_chips.png)  

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

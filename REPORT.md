# Method Description

The method is as follows:

## Unpack scRNA data
![scRNA data unpacking](mermaid/SCRNA_unpack.png) 

## Unpack UC9_I patches and genes

Patches are 32x32x3 portions of the histology image.  Patches and genes are unpacked from the dataset with this process:

![UC9_I_unpack_patches_and_genes](mermaid/UC9_I_unpack_patches_and_genes.png)

Patch locations are shown as black dots on the UC9_I image:

![UC9_I_patch_locations_on_image](mermaid/UC9_I_patch_locations_on_image.png)

The centroids are translated into spatial coordinates:

![UC9_I_spatial_coordinates](mermaid/UC9_I_spatial_coordinates.png)

Here are examples of some patches:

![UC9_I_patch_examples](mermaid/UC9_I_patch_examples.png)

## Template patches to features

This uses Resnet50 with 32x32x3 images upsized to 224x224 to extract image features.  This is used as a dimension reduction.

![template_patches_to_features](mermaid/template_patches_to_features.png)  

## UC9_I patches to features 

Apply patches to features pipeline to UC9_I patches.  Result is a 196937x1024 matrix of features. 

## Template principal component dimension reduction fit and transform pipeline

We will use PCA as a further dimension reduction on both patch features and gene expressions.  This is a template for the PCA dimension reduction process:

![template_pca_fit_transform](mermaid/template_pca_fit_transform.png)  

To determine the best number of principal components for our process, we look at several criteria:

* 90% **cumulative explained variance**
* **Elbow method**: Plot the eigenvalues and look for the "elbow" point where the marginal gain drops off using the python `kneed` package.
* **Kaiser criterion**: Keep components with eigenvalues > 1. Assumes standardized data.

In interest of getting things done quickly, we take the maximum of 2 and the minimum of the number of components reported by the cumulative variance, knee and Kaiser criteria.  

## UC9_I features to PCs

Apply principal component dimension reduction pipeline to features.

Final choice for UC9_I_feature number of principal components: 53.

![UC9_I_feature_density](mermaid/UC9_I_feature_density.png)

![UC9_I_feature_correlation_density](mermaid/UC9_I_feature_correlation_density.png)

![UC9_I_feature_correlation_matrix](mermaid/UC9_I_feature_correlation_matrix.png)

![UC9_I_feature_cumvar_method](mermaid/UC9_I_feature_cumvar_method.png)

![UC9_I_feature_elbow_method](mermaid/UC9_I_feature_elbow_method.png)

![UC9_I_feature_kaiser_method](mermaid/UC9_I_feature_kaiser_method.png)


## UC9_I Genes 460 to Genes 458

Our scRNA data only includes 458 of the 460 genes in the histology data, so we throw out 2 genes:

![UC9_I_genes460_to_genes458](mermaid/UC9_I_genes460_to_genes458.png)


## UC9_I Genes 458 to PCs

Apply PCA transform pipeline to UCI_9 458 gene expressions.  We choose 25 principal components.

![UC9_I_genes458_density](mermaid/UC9_I_genes458_density.png)

![UC9_I_genes458_correlation_density](mermaid/UC9_I_genes458_correlation_density.png)

![UC9_I_genes458_correlation_matrix](mermaid/UC9_I_genes458_correlation_matrix.png)

![UC9_I_genes458_cumvar_method](mermaid/UC9_I_genes458_cumvar_method.png)

![UC9_I_genes458_elbow_method](mermaid/UC9_I_genes458_elbow_method.png)

![UC9_I_genes458_kaiser_method](mermaid/UC9_I_genes458_kaiser_method.png)


## Template Ridge regression fit pipeline
![template_ridge_fit](mermaid/template_ridge_fit.png)

## UC9_I regress feature PCs to gene PCs
Apply Ridge regression fit pipeline to create a linear map from Feature PCs to Gene PCS.

![scRNA_genes458_PCs_scRNA_genes18157_PCs_ridge_fit_pred](mermaid/scRNA_genes458_PCs_scRNA_genes18157_PCs_ridge_fit_pred.png)

![scRNA_genes458_PCs_scRNA_genes18157_PCs_ridge_fit_spearman](mermaid/scRNA_genes458_PCs_scRNA_genes18157_PCs_ridge_fit_spearman.png)

## Template PCA transform pipeline
![template_pca_transform](mermaid/template_pca_transform.png)

## scRNA Genes 458 to UC9_I Genes 458 basis PCs
Apply PCA transform pipeline with UC9_I Genes 458 PCA basis to map scRNA Genes 458 to PCs.

![scRNA_genes458_density_pca_transform](mermaid/scRNA_genes458_density_pca_transform.png)

![scRNA_genes458_PCA_MSE](mermaid/scRNA_genes458_PCA_MSE.png)

## scRNA Genes 18157 to PCs
Apply PCA fit transform pipeline to scRNA 18157 gene expressions.  We choose 87 components.

![scRNA_genes18157_density](mermaid/scRNA_genes18157_density.png)
![scRNA_genes18157_correlation_matrix](mermaid/scRNA_genes18157_correlation_matrix.png)
![scRNA_genes18157_correlation_density](mermaid/scRNA_genes18157_correlation_density.png)
![scRNA_genes18157_elbow_method](mermaid/scRNA_genes18157_elbow_method.png)
![scRNA_genes18157_kaiser_method](mermaid/scRNA_genes18157_kaiser_method.png)
![scRNA_genes18157_cumvar_method](mermaid/scRNA_genes18157_cumvar_method.png)

## scRNA Regress Gene 458 PCs to Gene 18157 PCs
Apply Ridge regression pipeline to create a linear map from Gene 458 PCs to Gene 18157 PCS.

![scRNA_genes458_PCs_scRNA_genes18157_PCs_ridge_fit_pred](mermaid/scRNA_genes458_PCs_scRNA_genes18157_PCs_ridge_fit_pred.png)

![scRNA_genes458_PCs_scRNA_genes18157_PCs_ridge_fit_spearman](mermaid/scRNA_genes458_PCs_scRNA_genes18157_PCs_ridge_fit_spearman.png)

## UC9_I TIF Unpack

![UC9_I_tif_unpack](mermaid/UC9_I_tif_unpack.png)

## Dysplasia/Non-dysplasia patches to 1024 features
Apply patches to features pipeline to dysplasia/non-dysplasia patches.

## Dysplasia 1024 features to Feature PCs
Apply PCA transform pipeline to features using UC9_I Feature PCs basis.

![UC9_I_dysplasia_feature_density_pca_transform](mermaid/UC9_I_dysplasia_feature_density_pca_transform.png)

![UC9_I_dysplasia_feature_PCA_MSE](mermaid/UC9_I_dysplasia_feature_PCA_MSE.png)

## Non-dysplasia 1024 features to Feature PCs
Apply PCA transform pipeline to features using UC9_I Feature PCs basis.

![UC9_I_non_dysplasia_feature_density_pca_transform](mermaid/UC9_I_non_dysplasia_feature_density_pca_transform.png)

![UC9_I_non_dysplasia_feature_PCA_MSE](mermaid/UC9_I_non_dysplasia_feature_PCA_MSE.png)

## Template Ridge transform pipeline

![template_ridge_transform](mermaid/template_ridge_transform.png)

## Dysplasia/Non-dysplasia Feature PCs to 458 Gene PCs
Apply Ridge transform pipeline to feature PCs using UC9_I feature PCs to Gene PCs weights.

![scRNA_genes458_PCA_MSE](mermaid/scRNA_genes458_PCA_MSE.png)

## Dysplasia/Non-dysplasia 458 Gene PCs to 18157 Gene PCs
Apply Ridge transform pipeline to 458 Gene PCs using scRNA 458 PCs to 18157 PCs weights.

## Dysplasia 18157 Gene PCs to 18157 Genes
Apply inverse PCA transform pipeline to gene PCs using scRNA 18157 gene basis.

![UC9_I_dysplasia_genes18157_density_pca_inverse_transform](mermaid/UC9_I_dysplasia_genes18157_density_pca_inverse_transform.png)

## Non-dysplasia 18157 Gene PCs to 18157 Genes
Apply inverse PCA transform pipeline to gene PCs using scRNA 18157 gene basis.

![UC9_I_non_dysplasia_genes18157_density_pca_inverse_transform](mermaid/UC9_I_non_dysplasia_genes18157_density_pca_inverse_transform.png)

## Dysplasia 458 Gene PCs to 458 Genes
Apply inverse PCA transform pipeline to gene PCs using scRNA 18157 gene basis.

![UC9_I_dysplasia_genes458_density_pca_inverse_transform](mermaid/UC9_I_dysplasia_genes458_density_pca_inverse_transform.png)

## Non-dysplasia 458 Gene PCs to 458 Genes
Apply inverse PCA transform pipeline to gene PCs using scRNA 18157 gene basis.

![UC9_I_non_dysplasia_genes458_density_pca_inverse_transform](mermaid/UC9_I_non_dysplasia_genes458_density_pca_inverse_transform.png)

## Dysplasia/Non-dysplasia 458 Genes and 18157 Genes to 18615 Gene Expression
Concatenate.

## Genes ranked by highest absolute differential expression
![Crunch3 Processing](mermaid/crunch3.png)  

The following plot shows which impactful genes came from the original set of 460 and which came from imputation by Ridge regression:

![logFC_plot](mermaid/logFC_plot.png)

# Rationale

For the gene panel we suggest to take the top 500 genes in our sorted list which is ranked by absolute value of log fold change between dysplasia and non-dysplasia patches.

# Data and Resources Used

We use only the data provided by Broad Institute.  We do not use external data.  We use the registered images.  We don't do anything with the unregistered images.  We use the provided Crunch 3 dataset to select dysplasia and non-dysplasia cells for analysis.

We use the large dataset for UC9_I, the UC9_I tif files and the scRNA data.

We train on an Ubuntu PC with 1TB of SSD, an Intel Core i9 processor, 64GB of RAM, and an NVidia GTX 3060 with 12GB of VRAM.

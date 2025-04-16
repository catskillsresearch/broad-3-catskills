import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats import ttest_ind
import networkx as nx

def generate_bioinformatics_plots(df):
    # Ensure the dataframe is numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    # 1. Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df, cmap='viridis', cbar=True)
    plt.title('Heatmap of Gene Expression')
    plt.show()

    # 2. PCA Plot
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.T)
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    plt.title('PCA Plot')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    # 3. Volcano Plot
    if df.shape[0] > 1:
        group1, group2 = df.iloc[:df.shape[0]//2], df.iloc[df.shape[0]//2:]
        t_stat, p_values = ttest_ind(group1, group2, axis=0, nan_policy='omit')
        log2_fold_change = np.log2(group2.mean() / group1.mean())
        plt.figure(figsize=(8, 6))
        plt.scatter(log2_fold_change, -np.log10(p_values), alpha=0.7)
        plt.title('Volcano Plot')
        plt.xlabel('Log2 Fold Change')
        plt.ylabel('-Log10 P-value')
        plt.show()

    # 4. MA Plot
    mean_expression = (group1.mean() + group2.mean()) / 2
    log_ratio = np.log2(group2.mean() / group1.mean())
    plt.figure(figsize=(8, 6))
    plt.scatter(mean_expression, log_ratio, alpha=0.7)
    plt.title('MA Plot')
    plt.xlabel('Mean Expression')
    plt.ylabel('Log2 Fold Change')
    plt.show()

    # 5. Gene Expression Plot (Violin Plot for all genes)
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df)
    plt.title('Violin Plot for All Genes')
    plt.show()

    # 6. Network Map (Correlation-based)
    correlation_matrix = df.corr()
    G = nx.Graph()
    for gene1 in correlation_matrix.columns:
        for gene2 in correlation_matrix.columns:
            if gene1 != gene2 and correlation_matrix.loc[gene1, gene2] > 0.8:
                G.add_edge(gene1, gene2, weight=correlation_matrix.loc[gene1, gene2])
    plt.figure(figsize=(10, 8))
    nx.draw(G, with_labels=True, node_size=500, node_color='lightblue')
    plt.title('Network Map of Gene Correlations')
    plt.show()

    # 7. Dot Plot (Expression Summary)
    expression_summary = df.mean()
    plt.figure(figsize=(10, 6))
    plt.scatter(expression_summary.index, expression_summary.values, alpha=0.7)
    plt.title('Dot Plot of Gene Expression')
    plt.xlabel('Genes')
    plt.ylabel('Mean Expression')
    plt.xticks(rotation=90)
    plt.show()

if __name__=="__main__":
    data = np.random.rand(100, 10)  # 100 cells, 10 genes
    df = pd.DataFrame(data, columns=[f'Gene_{i}' for i in range(10)])
    generate_bioinformatics_plots(df)
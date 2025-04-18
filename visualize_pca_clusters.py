import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap  # Import UMAP

def visualize_pca_clusters(X, n_components):
    """
    Applies PCA, KMeans clustering, and visualizes the results using UMAP.

    Args:
        X (np.ndarray): Standardized data matrix (samples x features).
        n_components (int): Number of principal components and clusters to use.
    """
    # Apply PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_components, n_init=10)
    cluster_labels = kmeans.fit_predict(X_pca)

    # Visualize using UMAP for 2D projection
    reducer = umap.UMAP(n_components=2)
    X_umap = reducer.fit_transform(X_pca)

    # Create a scatter plot of the PCA-reduced data, colored by cluster labels
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_labels, cmap='viridis', s=5)
    plt.title(f'PCA-Reduced Data with KMeans Clusters (k={n_components})')
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')

    # Add a legend
    legend1 = plt.legend(*scatter.legend_elements(),
                         loc="upper right", title="Clusters")
    plt.gca().add_artist(legend1)

    plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import umap  # Import UMAP
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

def evaluate_clustering(X, n_components, method='pca'):
    """
    Applies either PCA or UMAP for initial dimension reduction, then KMeans clustering,
    and calculates Silhouette Score and Davies-Bouldin Index.

    Args:
        X (np.ndarray): Standardized data matrix (samples x features).
        n_components (int): Number of components to use.
        method (str): 'pca' or 'umap', specifying the dimension reduction method.

    Returns:
        dict: A dictionary containing Silhouette Score and Davies-Bouldin Index.
    """
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
    else:
        raise ValueError("Method must be 'pca' or 'umap'")

    # Clustering with KMeans
    kmeans = KMeans(n_clusters=n_components, n_init=10)
    cluster_labels = kmeans.fit_predict(X_reduced)

    # Evaluate clustering
    silhouette = silhouette_score(X_reduced, cluster_labels)
    db_index = davies_bouldin_score(X_reduced, cluster_labels)

    return {'Silhouette Score': silhouette, 'Davies-Bouldin Index': db_index, 'cluster_labels': cluster_labels, 'X_reduced': X_reduced}

def visualize_clusters(X, n_components, method='pca'):
    """
    Visualizes clusters after applying either PCA or UMAP for initial dimension reduction,
    then KMeans clustering. Adds centroid labels.

    Args:
        X (np.ndarray): Standardized data matrix (samples x features).
        n_components (int): Number of components to use.
        method (str): 'pca' or 'umap', specifying the dimension reduction method.
    """
    # Evaluate clustering to get cluster labels and reduced data
    eval_results = evaluate_clustering(X, n_components, method)
    cluster_labels = eval_results['cluster_labels']
    X_reduced = eval_results['X_reduced']
    silhouette = eval_results['Silhouette Score']
    db_index = eval_results['Davies-Bouldin Index']
    
    if method == 'pca':
        reducer = PCA(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
    elif method == 'umap':
        reducer = umap.UMAP(n_components=n_components)
        X_reduced = reducer.fit_transform(X)
    else:
        raise ValueError("Method must be 'pca' or 'umap'")

    # Clustering with KMeans
    kmeans = KMeans(n_clusters=n_components, n_init=10)
    cluster_labels = kmeans.fit_predict(X_reduced)

    # Visualize using UMAP for 2D projection
    reducer_umap = umap.UMAP(n_components=2)
    X_umap = reducer_umap.fit_transform(X_reduced)

    # Calculate centroids
    centroids = []
    for i in range(n_components):
        centroids.append(np.mean(X_umap[cluster_labels == i], axis=0))
    centroids = np.array(centroids)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 10))  # Increased figure size for better readability
    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=cluster_labels, cmap='viridis', s=5)
    ax.set_title(f'{method.upper()} (k={n_components})\nSilhouette={silhouette:.2f}, DBIndex={db_index:.2f}', fontsize=14)  # Increased fontsize for title
    ax.set_xlabel('UMAP Component 1', fontsize=12)  # Increased fontsize for labels
    ax.set_ylabel('UMAP Component 2', fontsize=12)
    
    # Add centroid labels
    for i, centroid in enumerate(centroids):
        ax.text(centroid[0], centroid[1], str(i + 1), color='magenta', fontsize=18, ha='center', va='center', weight='bold')

    # Add a legend
    legend1 = ax.legend(*scatter.legend_elements(), loc="upper right", title="Clusters")
    ax.add_artist(legend1)

    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    plt.show()
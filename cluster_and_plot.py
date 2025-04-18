import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Clustering methods
def cluster_and_plot(X, X_pca, n_clusters = 4):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    model = KMeans(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    title = 'KMeans Clustering'

    # Plot PCA results
    scatter = ax1.scatter(X_pca[:,0], X_pca[:,1], c=labels, cmap='viridis', alpha=0.7)
    ax1.set_title(f'PCA Projection\n{title}')
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    
    # Plot feature space (first 2 features)
    ax2.scatter(X[:,0], X[:,1], c=labels, cmap='viridis', alpha=0.7)
    ax2.set_title('Feature Space (First 2 Features)')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    plt.tight_layout()
    plt.show()
    
    return labels
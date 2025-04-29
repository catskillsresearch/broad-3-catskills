#!/usr/bin/env python
# coding: utf-8

# ## Debug pipeline

# ## Features to PCs

# ### Feature range and density

# In[1]:


get_ipython().system('pwd')


# In[15]:


get_ipython().system('ls resources/run/*feature* | grep -v dysp | grep -v ridge')


# In[48]:


from np_loadz import np_loadz
from select_random_from_2D_array import select_random_from_2D_array
import matplotlib.pyplot as plt
import os, gc, gzip, pickle, tempfile
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


# In[ ]:


data = np_loadz('resources/run/UC9_I_features.npz')


# In[18]:


data.shape


# In[19]:


data.min(), data.max()


# In[12]:


class foo:
    pass
self = foo()
self.object_type = 'feature'


# In[16]:


get_ipython().system('ls mermaid/*feature* | grep -v dysp | grep -v ridge')


# In[29]:


data_flat = select_random_from_2D_array(data, 1000)
plt.hist(data_flat, bins=100, density=True)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title(f"{self.object_type} density")
plt.savefig('mermaid/UC9_I_feature_density.png', dpi=150, bbox_inches='tight')


# ### Feature correlation

# In[127]:


import numpy as np
rho = np.corrcoef(data.T)


# The features appear from their correlation matrix to be uniformly uncorrelated.

# In[128]:


self.object_name = 'UC9_I'


# In[132]:


plt.imshow(rho)
plt.title('Resnet50 1024 feature correlation')
tag = f'{self.object_name}_{self.object_type}'
fn = f'mermaid/{tag}_correlation_matrix.png'
plt.savefig(fn, dpi=150, bbox_inches='tight')
fn


# In[131]:


fn


# In[140]:


rho[rho>= 0.999] = np.nan


# In[141]:


rhomin, rhomax = np.nanmin(rho), np.nanmax(rho)
rhomin, rhomax


# The distribution of feature correlation shows that off-diagonal correlations are in range [-0.66, 0.79].  So certain features have significant correlation.

# In[143]:


data_flat = select_random_from_2D_array(rho, 2000)
plt.hist(data_flat, bins=200, density=True)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title(f"Off-diagonal {self.object_type} correlation density in [{rhomin:.2f},{rhomax:.2f}]")
fn1 = f'mermaid/{tag}_correlation_density.png'
plt.savefig(fn1, dpi=150, bbox_inches='tight')
fn1


# ### Feature PCA

# #### Calc PCA

# In[55]:


self.sample_size = 100000


# In[56]:


# Create a generator for reproducibility
rng = np.random.default_rng()
# For a 2D array `arr` with shape (N, M)
sampled_rows = rng.choice(data.shape[0], size=self.sample_size, replace=False)  # Indices
sample = data[sampled_rows]

basis, scaler, L, V, MSE, pca_mean, finish = pca_analysis(sample, self.mse_goal, start=48, end = 50)
# In[59]:


X = sample

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Full PCA (compute ONCE)
pca = PCA(n_components=None)
pca.fit(X_scaled)


# #### Criterion: Cumulative explained variance

# In[135]:


explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
goal = 0.9
K_cumvar = np.argmax(explained_variance_ratio >= goal) + 1


# In[145]:


ftag = f'{tag}_pca_cumvar'
fn = f'mermaid/{ftag}.png'


# In[148]:


# Get basis and eigenvalues
B = pca.components_.T  # M x M basis
L = pca.explained_variance_/B.shape[0]
V = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8,6))
plt.plot(V)
plt.xlabel('Number of components')
plt.ylabel(f"{self.object_type} cumulative explained variance")
plt.title(f"{self.object_type} number of components to retain {goal*100:.0f}% variance: {K_cumvar}")
plt.axvline(K_cumvar, color='red', linestyle='--')  # Dashed red line
plt.plot(K_cumvar, goal, 'ko', markersize=8)  # Black dot at intersection
plt.grid(True)
plt.savefig(fn, dpi=150, bbox_inches='tight')


# In[151]:


print(f'![{ftag}]({fn})')


# #### Criterion: Scree Plot (Elbow Method)

# Plot the eigenvalues and look for the "elbow" point where the marginal gain drops off. This is typically done visually, but there are automated heuristics.

# In[ ]:


from kneed import KneeLocator
kl = KneeLocator(range(1, len(pca.explained_variance_)+1), pca.explained_variance_, curve='convex', direction='decreasing')
K_nee= kl.knee


# In[152]:


ftag = f'{tag}_scree'
fn = f'mermaid/{ftag}.png'


# In[153]:


plt.figure(figsize=(8,6))
plt.plot(np.arange(1, len(pca.explained_variance_)+1), pca.explained_variance_, marker='o')
plt.xlabel('Component number')
plt.ylabel('Eigenvalue')
plt.title(f"{self.object_type} Scree components to elbow: {K_nee}")
plt.axvline(K_nee, color='red', linestyle='--')  # Dashed red line
plt.grid(True)
plt.savefig(fn, dpi=150, bbox_inches='tight')
print(f'![{ftag}]({fn})')


# #### Criterion: Kaiser Criterion (Eigenvalues > 1)

# Keep components with eigenvalues > 1. Assumes standardized data.

# In[120]:


eigenvalues = pca.explained_variance_
K_aiser = np.sum(eigenvalues > 1)
print(f"Number of components with eigenvalue > 1: {K_aiser}")


# In[156]:


ftag = f'{tag}_kaiser'
fn = f'mermaid/{ftag}.png'


# In[157]:


plt.figure(figsize=(8,6))
plt.plot(np.arange(1, len(pca.explained_variance_)+1), pca.explained_variance_, marker='o')
plt.xlabel('Component number')
plt.ylabel('Eigenvalue')
plt.title(f"{self.object_type} Kaiser components to eigenvalue < 1: {K_aiser}")
plt.axvline(K_nee, color='red', linestyle='--')  # Dashed red line
plt.grid(True)
plt.savefig(fn, dpi=150, bbox_inches='tight')
print(f'![{ftag}]({fn})')


# #### Criterion: Parallel Analysis (compare to random data)

# Compare eigenvalues with those from randomly generated data. Retain components whose eigenvalues exceed the random counterparts.

# In[99]:


from sklearn.utils import resample

n_iter = 100
random_eigenvalues = np.zeros((n_iter, X.shape[1]))

for i in range(n_iter):
    X_random = np.random.normal(size=X.shape)
    pca_random = PCA().fit(X_random)
    random_eigenvalues[i, :] = pca_random.explained_variance_

random_mean = np.mean(random_eigenvalues, axis=0)

eigenvalues = pca.explained_variance_

K_parallel = np.sum(eigenvalues > random_mean)
print(f"Number of components exceeding random eigenvalues: {K_parallel}")


# ### Final criterion

# In interest of getting things done quickly, we will take the minimum number of components reported by the cumulative variance, knee and Kaiser criteria.  We will skip the Parallel Analysis criterion because it takes too long to compute for our data.

# In[137]:


K_final = max(2, min(K_cumvar, K_nee, K_aiser))
K_final


# In[160]:


print(f"Final choice for {tag} number of principal components: {K_final}.")


# In[ ]:





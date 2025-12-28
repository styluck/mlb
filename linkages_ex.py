# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 13:36:45 2024

@author: lich5
"""

import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, inconsistent, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data and standardize
data = pd.read_excel('examp4_2.xls', header=None)
X = data.iloc[1:, 1:].to_numpy()  # Exclude the first column (assumed labels)
obs_labels = data.iloc[:, 0].tolist()  # First column as labels
X = StandardScaler().fit_transform(X)  # Standardize the data

# Step 1: Perform hierarchical clustering using 'average' linkage
'''Cluster directly with maxclust = 3'''
from scipy.cluster.hierarchy import fclusterdata
cluster_labels = fclusterdata(X,3,criterion='maxclust',method = 'average')


# Print observations in each cluster
print("Cluster 1:", [obs_labels[i] for i in range(len(obs_labels)-1) if cluster_labels[i] == 1])
print("Cluster 2:", [obs_labels[i] for i in range(len(obs_labels)-1) if cluster_labels[i] == 2])
print("Cluster 3:", [obs_labels[i] for i in range(len(obs_labels)-1) if cluster_labels[i] == 3])

# Step 2: Perform hierarchical clustering step by step
# Compute pairwise distances
y = pdist(X)

'''Compute the linkage matrix'''
Z = linkage(y, method='average')

# Plot the dendrogram
plt.figure(figsize=(10, 8))
plt.rcParams['font.family'] = 'SimHei' # SimSun
dendrogram(
    Z,
    orientation='right',
    labels=obs_labels[1:],
    color_threshold=0,  # Single color for all branches
)
plt.xlabel('Standardized Distance (Average Linkage)')
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# Compute inconsistency coefficients (for a threshold of 40)
inconsistencies = inconsistent(Z, d=40)
print("Inconsistency coefficients (threshold 40):\n", inconsistencies)

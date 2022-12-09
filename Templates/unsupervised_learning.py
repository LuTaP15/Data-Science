"""
Template for typical Unsupervised Learning tasks

1. PCA
2. kMeans
"""

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# PCA
pca = PCA(n_components=2).fit(X)
X_pca = pca.fit_transform(X)

# PCA plot
plt.figure(figsize=(20,15))
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.show()

# PCA plot with label
plt.figure(figsize=(20,15))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.show()

#####################################################################################################################
# KMeans
import numpy as np
from sklearn.cluster import KMeans

# Elbow Method
num_clusters_to_try = 20
inertias = []
for clusters in range(1,num_clusters_to_try):
  kmeans = KMeans(n_clusters=clusters).fit(X_pca)
  inertias.append(kmeans.inertia_)

plt.plot(np.arange(1, num_clusters_to_try), inertias)


# K-Means
kmeans = KMeans(n_clusters=4, random_state=1).fit(X_pca)
labels = kmeans.predict(X_pca)

plt.scatter(X_pca[:,0], X_pca[:,1], c=labels)
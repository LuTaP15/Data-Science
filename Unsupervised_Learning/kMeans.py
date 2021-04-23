"""
This is an example implementation of kMeans.
KMeans is used for clustering.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

######################################################################################################################
# Einlesen der Daten
df = pd.read_csv("./../data/autos_prepared.csv")

X = df[["yearOfRegistration", "price"]]

# Skalieren der Daten
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)

# Model erstellen mit n_clusters
model = KMeans(n_clusters=3)
model.fit(X_transformed)

# Labels für die Datenpunkte
#print(model.labels_)

# Plot der Cluster und deren Centroids
labels = model.labels_
centers = model.cluster_centers_

# Rückskalierung für den Plot
centers_transformed = scaler.inverse_transform(centers)

plt.scatter(df["yearOfRegistration"], df["price"],
            c=labels, s=10)
plt.scatter(centers_transformed[:, 0],
            centers_transformed[:, 1],
            c=range(len(centers_transformed)),
            marker='X', s=100)
plt.show()
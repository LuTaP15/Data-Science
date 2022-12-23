"""
This program applies DBSCAN clustering to the autos_prepared.csv dataset and visualizes the resulting clusters.
It reads in and processes the dataset, creates and fits a DBSCAN model, and plots the results.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

######################################################################################################################
# Read in data
df = pd.read_csv("../../data/autos_prepared.csv")

# Select features for clustering
X = df[["yearOfRegistration", "price"]]

# Scale data
scaler = StandardScaler()
X_transformed = scaler.fit_transform(X)

# Create DBSCAN model
model = DBSCAN(eps=0.3, min_samples=5)
model.fit(X_transformed)

# Labels for data points
#print(model.labels_)

# Plot clusters and their labels
labels = model.labels_

plt.scatter(df["yearOfRegistration"], df["price"],
            c=labels, s=10)
plt.show()


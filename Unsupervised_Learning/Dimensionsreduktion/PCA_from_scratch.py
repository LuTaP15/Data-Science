"""
PCA = Principal Component Analysis
"""
# Libaries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("./../../data/iris_data.csv")

# Check the loaded data
print(df.head())

# Take the label to Y
Y = df.pop("species")
# Take all features to X
X = df

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate Covariance matrix
cov_matrix = np.cov(X_scaled.T)

# Eigendecomposition
eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)

# Calculate the explained variance by the principal components
explained_variance = []
for i in range(len(eigen_values)):
    explained_variance.append(eigen_values[i]/np.sum(eigen_values))

sum_ev = np.sum(explained_variance)
print(f"Sum of explained variance by all principal components:\n {sum_ev}")
print(f"List of the explained variance by principal component:\n {explained_variance}")

# Visualization
# Create the two principal components
pc_1 = X_scaled.dot(eigen_vectors.T[0])
pc_2 = X_scaled.dot(eigen_vectors.T[1])

# Create a new dataframe with two principal components
res = pd.DataFrame(zip(pc_1, pc_2, Y), columns=["PC1", "PC2", "Label"])

# 1D Plot
plt.figure(figsize=(16, 10))
sns.scatterplot(x=res["PC1"], y=[0]*len(res), hue=res["Label"], s=100)
plt.show()

# 2D Plot
plt.figure(figsize=(16, 10))
sns.scatterplot(x=res["PC1"], y=res["PC2"], hue=res["Label"], s=100)
plt.show()

# 3D Plot
pc_3 = X_scaled.dot(eigen_vectors.T[2])
res["PC3"] = pc_3

fig = px.scatter_3d(df, x=res["PC1"], y=res["PC2"], z=res["PC3"],
              color=res["Label"])
fig.show()
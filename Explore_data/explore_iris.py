"""
This file shows different techniques for exploring data based on the iris dataset.
"""

import numpy as np
import pandas as pd
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# Load the iris dataset
iris_df = pd.read_csv("./../data/iris_data.csv")

# Print the first 5 entries of the dataset
print(iris_df.head())

# Nummerical overview
print(iris_df.describe())

# Distribution of the dataset
print(iris_df.groupby('species').size())

# Split the dataset into train and test
train, test = train_test_split(iris_df, test_size=0.4, stratify=iris_df["species"], random_state=42)

########################################################################################################
# Plotting

# Histogram
n_bins = 10
fig, axs = plt.subplots(2, 2)
axs[0,0].hist(train['sepal_length'], bins=n_bins)
axs[0,0].set_title('Sepal Length')
axs[0,1].hist(train['sepal_width'], bins=n_bins)
axs[0,1].set_title('Sepal Width')
axs[1,0].hist(train['petal_length'], bins=n_bins)
axs[1,0].set_title('Petal Length')
axs[1,1].hist(train['petal_width'], bins=n_bins)
axs[1,1].set_title('Petal Width')
# add some spacing between subplots
fig.tight_layout(pad=1.0)
plt.show()

# Boxplot
fig, axs = plt.subplots(2, 2)
fn = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
cn = ['setosa', 'versicolor', 'virginica']
sns.boxplot(x='species', y='sepal_length', data=train, order=cn, ax=axs[0,0])
sns.boxplot(x='species', y='sepal_width', data=train, order=cn, ax=axs[0,1])
sns.boxplot(x='species', y='petal_length', data=train, order=cn, ax=axs[1,0])
sns.boxplot(x='species', y='petal_width', data=train,  order=cn, ax=axs[1,1])
# add some spacing between subplots
fig.tight_layout(pad=1.0)
plt.show()

# Violinplot
sns.violinplot(x="species", y="petal_width", data=train, size=5, order=cn, palette='colorblind')
plt.show()

# Scatterplot
sns.pairplot(train, hue="species", height=2, palette='colorblind')
plt.show()

# Correlationmatrix
corrmat = train.corr()
sns.heatmap(corrmat, annot=True, square=True)
plt.show()

# Parallel coordinates plot
parallel_coordinates(train, "species", color = ['blue', 'red', 'green'])
plt.show()
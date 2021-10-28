"""
Uniform Manifold Approximation and Projection (UMAP)

Package umap-learn

Projection:
the process or technique of reproducing a spatial object upon a plane, a curved surface, or a line by projecting its
points. You can also think of it as a mapping of an object from high-dimensional to low-dimensional space.
Approximation:
the algorithm assumes that we only have a finite set of data samples (points), not the entire set that makes up the
manifold. Hence, we need to approximate the manifold based on the data available.
Manifold:
a manifold is a topological space that locally resembles Euclidean space near each point. One-dimensional manifolds
include lines and circles, but not figure eights. Two-dimensional manifolds (a.k.a. surfaces) include planes,
spheres, torus, and more.
Uniform:
the uniformity assumption tells us that our data samples are uniformly (evenly) distributed across the manifold.
In the real world, however, this is rarely the case. Hence, this assumption leads to the notion that the distance
varies across the manifold. i.e., the space itself is warping: stretching or shrinking according to where the
data appear sparser or denser.
"""
# Data manipulation
import pandas as pd # for data manipulation
import numpy as np # for data manipulation

# Visualization
import plotly.express as px # for data visualization
import matplotlib.pyplot as plt # for showing handwritten digits

# Skleran
from sklearn.datasets import load_digits # for MNIST data
from sklearn.model_selection import train_test_split # for splitting data into train and test samples

# UMAP dimensionality reduction
from umap import UMAP

# Load digits data
digits = load_digits()

# Load arrays containing digit data (64 pixels per image) and their true labels
X, y = load_digits(return_X_y=True)

# Some stats
print('Shape of digit images: ', digits.images.shape)
print('Shape of X (main data): ', X.shape)
print('Shape of y (true labels): ', y.shape)

# Display images of the first 10 digits
fig, axs = plt.subplots(2, 5, sharey=False, tight_layout=True, figsize=(12,6), facecolor='white')
n=0
plt.gray()
for i in range(0,2):
    for j in range(0,5):
        axs[i,j].matshow(digits.images[n])
        axs[i,j].set(title=y[n])
        n=n+1
plt.show()


def chart(X, y):
    # --------------------------------------------------------------------------#
    # This section is not mandatory as its purpose is to sort the data by label
    # so, we can maintain consistent colors for digits across multiple graphs

    # Concatenate X and y arrays
    arr_concat = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
    # Create a Pandas dataframe using the above array
    df = pd.DataFrame(arr_concat, columns=['x', 'y', 'z', 'label'])
    # Convert label data type from float to integer
    df['label'] = df['label'].astype(int)
    # Finally, sort the dataframe by label
    df.sort_values(by='label', axis=0, ascending=True, inplace=True)
    # --------------------------------------------------------------------------#

    # Create a 3D graph
    fig = px.scatter_3d(df, x='x', y='y', z='z', color=df['label'].astype(str), height=900, width=950)

    # Update chart looks
    fig.update_layout(title_text='UMAP',
                      showlegend=True,
                      legend=dict(orientation="h", yanchor="top", y=0, xanchor="center", x=0.5),
                      scene_camera=dict(up=dict(x=0, y=0, z=1),
                                        center=dict(x=0, y=0, z=-0.1),
                                        eye=dict(x=1.5, y=-1.4, z=0.5)),
                      margin=dict(l=0, r=0, b=0, t=0),
                      scene=dict(xaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                 yaxis=dict(backgroundcolor='white',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            ),
                                 zaxis=dict(backgroundcolor='lightgrey',
                                            color='black',
                                            gridcolor='#f0f0f0',
                                            title_font=dict(size=10),
                                            tickfont=dict(size=10),
                                            )))
    # Update marker size
    fig.update_traces(marker=dict(size=3, line=dict(color='black', width=0.1)))

    fig.show()

##########################################################################################

# Configure UMAP hyperparameters
reducer = UMAP(n_neighbors=100,
               n_components=3,
               metric='euclidean',
               n_epochs=1000,
               learning_rate=1.0,
               init='spectral',
               random_state=42)

# Fit and transform the data
X_trans = reducer.fit_transform(X)

# Check the shape of the new data
print('Shape of X_trans: ', X_trans.shape)


##############################################################################################
# Plot transformed data
chart(X_trans, y)

##############################################################################################

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# Configure UMAP hyperparameters
reducer2 = UMAP(n_neighbors=100, n_components=3, n_epochs=1000,
                min_dist=0.5, local_connectivity=2, random_state=42,
              )

# Training on MNIST digits data - this time we also pass the true labels to a fit_transform method
X_train_res = reducer2.fit_transform(X_train, y_train)

# Apply on a test set
X_test_res = reducer2.transform(X_test)

# Print the shape of new arrays
print('Shape of X_train_res: ', X_train_res.shape)
print('Shape of X_test_res: ', X_test_res.shape)

##############################################################################################
# Plot train set
chart(X_train_res, y_train)
# Plot test set
chart(X_test_res, y_test)
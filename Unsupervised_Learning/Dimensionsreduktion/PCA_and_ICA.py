import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA


def plot_results(models, names):
    """
    Plot the results of the analysis.
    """
    colors = ['red', 'steelblue']
    fig, axes = plt.subplots(nrows=len(models), sharex=True, figsize=(8, 10))
    fig.suptitle('Independent Component Analysis Example')
    for i, (model, name) in enumerate(zip(models, names)):
        for sig, color in zip(model.T, colors):
            axes[i].plot(sig, color=color)
        axes[i].set_title(name)
    plt.tight_layout()
    plt.show()


# Generate a synthetic dataset with two independent sources
np.random.seed(42)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time) # Source 1: sine wave
s2 = np.sign(np.sin(3 * time)) # Source 2: square wave

S = np.c_[s1, s2]
S += 0.2 * np.random.normal(size=S.shape) # Add noise to the sources
S /= S.std(axis=0) # Standardize the sources

# Mix the sources together to create a mixed signal
A = np.array([[0.5, 0.5], [0.2, 0.8]]) # Mixing matrix
X = np.dot(S, A.T) # Mixed signal

# Standardize the data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Use PCA to reduce the dimensionality of the data
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Use ICA to separate the sources from the mixed signal
ica = FastICA(n_components=2)
X_ica = ica.fit_transform(X) # Estimated sources

# Plot the results
models = [X, S, X_pca, X_ica]
names = ['Observations (mixed signal)', 'True Sources', 'PCA features', 'ICA features']
plot_results(models, names)

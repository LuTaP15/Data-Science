import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris_dataset = load_iris()

print(f"Schlüssel von iris_dataset: {iris_dataset.keys()}")

print(f"Zielbezeihnungen: {iris_dataset['target_names']}")
print(f"Namen der Merkmale: {iris_dataset['feature_names']}")
print(f"Zielwerte: {iris_dataset['target']}")


X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# Daten anschauen

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
"""
grr = pd.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60,
                        alpha=.8)#, cmap=mglearn.cm3)
plt.show()
"""


# Model lernen
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Evaluieren 1
X_new = np.array([[5, 2.9, 1, 0.2]])
prediction = knn.predict(X_new)
print(f"Vorhersage: {prediction}, {iris_dataset['target_names'][prediction]}")

# Evaluieren 2
y_pred = knn.predict(X_test)
print(f"Vorhersagen für den Testdatensatz: \n {y_pred}")
print(f"Genauigkeit auf den Testdaten:", np.mean(y_pred == y_test))
print(f"Genauigkeit auf den Testdaten:", knn.score(X_test, y_test))

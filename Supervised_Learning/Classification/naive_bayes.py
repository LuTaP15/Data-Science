"""
This is an example implementation of the naive bayes classifier based on the famous iris dataset
"""
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load iris data
data = load_iris()

X, y, column_names = data["data"], data["target"], data["feature_names"]
X = pd.DataFrame(X, columns=column_names)

# Split the data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=21)

# Train the model
model = GaussianNB()
model.fit(X_train, y_train)

# Calculate the accuracy score
score = accuracy_score(y_test, model.predict(X_test))

print(f"Genauigkeit ist {score:.10f}")



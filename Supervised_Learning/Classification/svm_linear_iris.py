"""
Support Vector Classifier for the iris dataset with a pipeline
"""

# Import the necessary libraries and modules
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the dataset and split it into training and testing sets
data = pd.read_csv("./../../data/iris_data.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='linear'))
])

# Train the pipeline on the training set
pipeline.fit(X_train, y_train)

# Evaluate the pipeline on the testing set
y_pred = pipeline.predict(X_test)
accuracy = pipeline.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

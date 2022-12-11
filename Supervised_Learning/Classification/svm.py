"""
Support Vector Machine for the wine data set
"""

# Import the necessary libraries and modules
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Load the wine dataset
data = load_wine()
X = data['data']
y = data['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])

# Define the hyperparameter grid
param_grid = {
    'classifier__kernel': ['linear', 'rbf'],
    'classifier__C': [0.1, 1, 10, 100]
}

# Create the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')

# Train the grid search object on the training set
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and evaluation score
print(grid_search.best_params_)
print(grid_search.best_score_)

# Evaluate the best model on the testing set
y_pred = grid_search.predict(X_test)
accuracy = grid_search.score(X_test, y_test)
print(f'Accuracy: {accuracy}')

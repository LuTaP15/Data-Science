"""
Building a Random Forest from scratch based on the Decision Tree from scratch
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# Import the Decision Tree from scratch
from decision_tree_from_scratch import DecisionTree



class RandomForest:

    def __init__(self, num_trees=5, subsample_size=None, max_depth=None, max_features=None, bootstrap=True, random_state=None):
        self.num_trees = num_trees
        self.subsample_size = subsample_size
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        # Will store individually trained decision trees
        self.decision_trees = []

    def fit(self, X, y):
        # Reset
        if len(self.decision_trees) > 0:
            self.decision_trees = []

        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(y, pd.core.series.Series):
            y = y.values

        # Build each tree of the forest
        num_built = 0

        while num_built < self.num_trees:

            clf = DecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,
                random_state=self.random_state
            )

            # Obtain data sample
            _X, _y = self.sample(X, y, self.random_state)
            # Train
            clf.fit(_X, _y)
            # Save the classifier
            self.decision_trees.append(clf)

            num_built += 1

            if self.random_state is not None:
                self.random_state += 1

    def sample(self, X, y, random_state):

        n_rows, n_cols = X.shape

        # Sample with replacement
        if self.subsample_size is None:
            sample_size = n_rows
        else:
            sample_size = int(n_rows*self.subsample_size)

        np.random.seed(random_state)
        samples = np.random.choice(a=n_rows, size=sample_size, replace=self.bootstrap)

        return X[samples], y[samples]

    def predict(self, X):
        # Make predictions with every tree in the forest
        y = []
        for tree in self.decision_trees:
            y.append(tree.predict(X))

        # Reshape so we can find the most common value
        y = np.swapaxes(y, axis1=0, axis2=1)

        # Use majority voting for the final prediction
        predicted_classes = stats.mode(y, axis=1)[0].reshape(-1)

        return predicted_classes


#####################################################################################################################
# Example

if __name__ == "__main__":
    # Load data
    iris_dataset = load_iris()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

    # Create and train new Decision Tree
    rf = RandomForest()
    rf.fit(X_train, y_train)


    # Evaluieren
    X_new = np.array([[5, 2.9, 1, 0.2]])
    prediction = rf.predict(X_new)
    print(f"Vorhersage: {prediction}, {iris_dataset['target_names'][prediction]}")


    y_pred = rf.predict(X_test)
    print(f"Vorhersagen für den Testdatensatz: \n {y_pred}")
    print(f"Genauigkeit auf den Testdaten:", np.mean(y_pred == y_test))
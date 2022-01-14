"""
Building a decision tree algorithm from scratch

"""

# Libaries
import numpy as np
import pandas as pd
import random
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def calculate_gini(a, b):
    gini = 1.0 - sum((n / a) ** 2 for n in b)
    return gini

# Create Class Node
class Node:
    def __init__(self, predicted_class):
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


# Create Decision Tree Class
class DecisionTree:
    def __init__(self, max_depth=None, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_state = random_state
        self.tree = None

    def fit(self, X, y):
        # Store number of classes and features of the dataset
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(y, pd.core.frame.Series):
            y = y.values


        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        if self.max_features == None:
            self.max_features = self.n_features


        if isinstance(self.max_features, float) and self.max_features<=1:
            self.max_features = int(self.max_features*self.n_features)

        # Create a tree
        self.tree = self.grow_tree(X, y, self.random_state)

    def grow_tree(self, X, y, random_state, depth=0):
        num_samples_per_class = [np.sum(y==i) for i in range(self.n_classes)]
        predicted_class = np.argmax(num_samples_per_class)

        node = Node(predicted_class=predicted_class)

        if (self.max_depth is None) or (depth < self.max_depth):
            id, thr = self.best_split(X, y, random_state)

            if id is not None:
                if random_state is not None:
                    random_state += 1

                indices_left = X[:, id] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]

                node.feature_index = id
                node.threshold = thr
                node.left = self.grow_tree(X_left, y_left, random_state, depth+1)
                node.right = self.grow_tree(X_right, y_right, random_state, depth + 1)

        return node

    def best_split(self, X, y, random_state):

        m = len(y)
        if m <= 1:
            return None, None

        num_class_parent = [np.sum(y==c) for c in range(self.n_classes)]
        #best_gini = 1.0 - sum((n/m)**2 for n in num_class_parent)
        best_gini = calculate_gini(m, num_class_parent)
        if best_gini == 0:
            return None, None

        best_feat_id, best_threshold = None, None
        random.seed(random_state)
        feat_indices = random.sample(range(self.n_features), self.max_features)

        for feat_id in feat_indices:
            sorted_column = sorted(set(X[:, feat_id]))
            threshold_values = [np.mean([a, b]) for a, b in zip(sorted_column, sorted_column[1:])]

            for threshold in threshold_values:
                left_y = y[X[:, feat_id] < threshold]
                right_y = y[X[:, feat_id] > threshold]

                num_class_left = [np.sum(left_y==c) for c in range(self.n_classes)]
                num_class_right = [np.sum(right_y == c) for c in range(self.n_classes)]

                gini_left = calculate_gini(len(left_y), num_class_left)
                gini_right = calculate_gini(len(right_y), num_class_right)

                gini = (len(left_y)/m)*gini_left + (len(right_y)/m)*gini_right

                if gini < best_gini:
                    best_gini = gini
                    best_feat_id = feat_id
                    best_threshold = threshold

        return best_feat_id, best_threshold

    def predict(self, X):
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values

        predicted_classes = np.array([self.predict_example(inputs) for inputs in X])

        return predicted_classes

    def predict_example(self, inputs):
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right

        return node.predicted_class


#####################################################################################################################
# Example

if __name__ == "__main__":
    # Load data
    iris_dataset = load_iris()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

    # Create and train new Decision Tree
    dt = DecisionTree()
    dt.fit(X_train, y_train)


    # Evaluieren
    X_new = np.array([[5, 2.9, 1, 0.2]])
    prediction = dt.predict(X_new)
    print(f"Vorhersage: {prediction}, {iris_dataset['target_names'][prediction]}")


    y_pred = dt.predict(X_test)
    print(f"Vorhersagen für den Testdatensatz: \n {y_pred}")
    print(f"Genauigkeit auf den Testdaten:", np.mean(y_pred == y_test))
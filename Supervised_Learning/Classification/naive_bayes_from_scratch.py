"""
Naive Bayes Classifier from scratch

Implementation of a Naive Bayes Classifier class and evaluation of it.
"""
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

############################################################################
# Build the Naive Bayes Classifier from scratch


class NaiveBayesClf:
    def fit(self, X, y):
        # Get number of samples (rows) and features (columns)
        self.n_samples, self.n_features = X.shape
        # Get number of unique classes
        self.n_classes = len(np.unique(y))
        # Create matrices to store mean, variance and prior
        self.mean = np.zeros((self.n_classes, self.n_features))
        self.variance = np.zeros((self.n_classes, self.n_features))
        self.priors = np.zeros(self.n_classes)

        for c in range(self.n_classes):
            # Create a subset fpr the specific class "c"
            X_c = X[y==c]
            # Calculate stats and update matrices
            self.mean[c,:] = np.mean(X_c, axis=0)
            self.variance[c,:] = np.var(X_c, axis=0)
            self.priors[c] = X_c.shape[0] / self.n_samples

    def predict(self, X):
        # For each sample x in dataset X
        y_hat = [self.get_class_probability(x) for x in X]
        return np.array(y_hat)

    def get_class_probability(self, x):
        posteriors = []

        for c in range(self.n_classes):
            # Get summary stats and prior
            mean = self.mean[c]
            variance = self.variance[c]
            prior = np.log(self.priors[c])

            # Calculate new posterior and append to list
            posterior = np.sum(np.log(self.gaussian_density(x, mean, variance)))
            posterior += prior
            posteriors.append(posterior)

        # Return the index with highest class probability
        return np.argmax(posteriors)

    def gaussian_density(self, x, mean, var):
        const = 1/np.sqrt(2*var*np.pi)
        proba = np.exp(-.5*((x-mean)**2/var))
        return const * proba

############################################################################
# Using the Naive Bayes Classifier on iris dataset

# Helper function
def get_accuracy(y_true, y_hat):
    return np.sum(y_true==y_hat) / len(y_true)


# Load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate, train and predict
nb = NaiveBayesClf()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)

# Results
res = get_accuracy(y_test, predictions)
print(f"Naive Bayes Classifier Accuracy: {res}")
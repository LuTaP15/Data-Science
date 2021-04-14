"""
This is an example implementation of logistic regression to classify numbers in the digits dataset.
1797 images of numbers from 0 to 9.
"""

# Libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

# Load dataset
digits = load_digits()

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=21)

# Create model
model = LogisticRegression()

# Skalieren der Daten
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Model fitten
model.fit(X_train, y_train)

# Berechnen des Scores auf den Testdaten
score = model.score(X_test, y_test)
print(f"Genauigkeit: {score*100:.5f}%")

# Confusionmatrix
cm = metrics.confusion_matrix(y_test, model.predict(X_test))

# Plot der Confusionmatrix
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, square=True, cmap="autumn_r")
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_title = f'Accuracy Score: {score*100:.5f}%'
plt.title(all_sample_title, size=15)
plt.show()


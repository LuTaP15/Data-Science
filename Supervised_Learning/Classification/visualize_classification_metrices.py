"""
Easy and quick visualisations with scikit-plot for classification metrices.

1. Confusion matrix
2. Receiver Operating Characteristic (ROC) Curve
3. Precision Recall Curve (PR Curve)
4. Calibration Plot
5. Cumulative Gains Curve
6. Lift Curve

1.
Confusion matrix compares the ground truth to the predicted label and classifies the results
into True Positive, True Negative, False Positive and False Negative.
2.
An ROC curve shows the performance (True Positive Rate aka Recall and False Positive Rate)
of a classifier at all classification thresholds against a random baseline classifier.
3.
A precision and recall curve shows precision and recall values at all classification thresholds.
It summarizes the trade off between precision and recall.
4.
Calibration plots is a diagnostic method to check if the predicted value
can directly be interpreted as confidence level.
5.
Cumulative gains curve shows the performance of a model and compares against a random baseline classifier.
6.
The lift curve shows the response rate for each class compared to a random baseline classifier
when considering a portion of the population with the highest probability.

"""


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_predict, train_test_split
import scikitplot as skplt
import matplotlib.pyplot as plt

# Load data
X, y = datasets.load_breast_cancer(return_X_y=True)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Create Gradient Boosting Classifier
gbc = GradientBoostingClassifier()
gbc_model = gbc.fit(X_train, y_train)
y_gbc_proba = gbc_model.predict_proba(X_test)
y_gbc_pred = np.where(y_gbc_proba[:,1] > 0.5, 1, 0)

# Create Random Forest Classifier
rfc = RandomForestClassifier()
rfc_model = rfc.fit(X_train, y_train)
y_rfc_proba = rfc_model.predict_proba(X_test)
y_rfc_pred = np.where(y_rfc_proba[:,1] > 0.5, 1, 0)


# Plot Confusion matrix
skplt.metrics.plot_confusion_matrix(y_test, y_gbc_pred, normalize=False, title = 'Confusion Matrix for GBC')
plt.show()

# Plot normalized Confusion matrix
skplt.metrics.plot_confusion_matrix(y_test, y_gbc_pred, normalize=True, title = 'Confusion Matrix for GBC')
plt.show()

# Receiver Operating Characteristic (ROC) Curve
skplt.metrics.plot_roc(y_test, y_gbc_proba, title = 'ROC Plot for GBC')
plt.show()

# Precision Recall Curve (PR Curve)
skplt.metrics.plot_precision_recall(y_test, y_gbc_proba, title = 'PR Curve for GBC')
plt.show()

# Calibration Plot
probas_list = [y_gbc_proba, y_rfc_proba]
clf_names = ['GBC', 'RF']
skplt.metrics.plot_calibration_curve(y_test, probas_list = probas_list, clf_names = clf_names)
plt.show()

# Cumulative Gains Curve
skplt.metrics.plot_cumulative_gain(y_test, y_gbc_proba, title = 'Cumulative Gains Chart for GBC')
plt.show()

# Lift Curve
skplt.metrics.plot_lift_curve(y_test, y_gbc_proba, title = 'Lift Curve for GBC')
plt.show()

# 2 in 1 plot
fig, ax = plt.subplots(1,2)
skplt.metrics.plot_cumulative_gain(y_test, y_gbc_proba, ax = ax[0], title = 'Cumulative Gains Chart for GBC')
skplt.metrics.plot_lift_curve(y_test, y_gbc_proba, ax = ax[1],  title = 'Lift Curve for GBC')
plt.show()


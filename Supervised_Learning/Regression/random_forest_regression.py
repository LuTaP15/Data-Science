"""
Beispiel Implementierung einer Random Forest Regression mit Daten von Häusern in Kalifornien

This dataset consists of 20,640 samples and 9 features.

feature_names:
"MedInc" = average income
"HouseAge" = housing average age
"AveRooms" = average rooms
"AveBedrms" = average bedrooms
"Population" = population
"AveOccup" = average occupation
"Latitude" = latitude
"Longitude" = longitude

target_names:
"MedHouseVal" = average house value

The original database is available from StatLib
http://lib.stat.cmu.edu/datasets/

References
----------
Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
Statistics and Probability Letters, 33 (1997) 291-297.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_error
from yellowbrick.regressor import ResidualsPlot, prediction_error

data = fetch_california_housing()

X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.25)

rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)
rf_reg.score(X_test, y_test)

y_pred = rf_reg.predict(X_test)

############################################################
# Evaluation
print(f"Metrics for Regression")
r2 = r2_score(y_test, y_pred)
print(f"R Square: {r2}")
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Square Error: {mse}")
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"Root Mean Square Error: {rmse}")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolut Error: {mae}")
evs = explained_variance_score(y_test, y_pred)
print(f"Explained variance score: {evs}")


############################################################
# Plot predicted vs true label
vis = prediction_error(rf_reg, X_train, y_train, X_test, y_test)

#############################################################
# Residual plot
visualizer = ResidualsPlot(rf_reg, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure



from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Generate data
X, Y = make_classification(n_samples=200, n_classes=2, n_features=10, n_redundant=0, random_state=42)

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# Random Forest Classifier
rf = RandomForestClassifier(max_features=5, n_estimators=100)

rf.fit(X_train, Y_train)
Y_pred = rf.predict(X_test)

# Evaluation
ac = accuracy_score(Y_pred, Y_test)
mcc = matthews_corrcoef(Y_pred, Y_test)
roc_auc = roc_auc_score(Y_pred, Y_test)

print(f"Accuracy: {ac} \n"
      f"Matthews Correlation Coefficient: {mcc} \n"
      f"Area Under the Receiver Operating Characteristic Curve: {roc_auc}")


# Hyperparametertuning
max_features_range = np.arange(1, 6, 1)
n_estimators_range = np.arange(10, 210, 10)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

rf = RandomForestClassifier()

grid = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='roc_auc', cv=5)

grid.fit(X_train, Y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

# Visualize Hyperparameters
grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),
                          pd.DataFrame(grid.cv_results_["mean_test_score"],
                          columns=["ROC_AUC"])],
                          axis=1)


grid_contour = grid_results.groupby(['max_features','n_estimators']).mean()


grid_reset = grid_contour.reset_index()
grid_reset.columns = ['max_features', 'n_estimators', 'ROC_AUC']
grid_pivot = grid_reset.pivot('max_features', 'n_estimators')


x = grid_pivot.columns.levels[1].values
y = grid_pivot.index.values
z = grid_pivot.values

# 2D-Plot
layout = go.Layout(
            xaxis=go.layout.XAxis(
              title=go.layout.xaxis.Title(
              text='n_estimators')
             ),
             yaxis=go.layout.YAxis(
              title=go.layout.yaxis.Title(
              text='max_features')
            ) )

fig = go.Figure(data=[go.Contour(z=z, x=x, y=y)], layout=layout )

fig.update_layout(title='Hyperparameter tuning', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()

# 3D-Plot
fig = go.Figure(data=[go.Surface(z=z, y=y, x=x)], layout=layout )

fig.update_layout(title='Hyperparameter tuning',
                  scene = dict(
                    xaxis_title='n_estimators',
                    yaxis_title='max_features',
                    zaxis_title='ROC_AUC'),
                  autosize=False,
                  width=800, height=800,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()
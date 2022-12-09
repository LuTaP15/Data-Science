"""
Template for typical Supervised Learning tasks

1. Models
2. Grid Search
3. Model learning
4. Evaluation Metrices
5. Logistic and Linear Regression with statsmodels
6. Feature Importance
"""

# imports
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, LogisticRegression
from xgboost import XGBRegressor, XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

# instantiate
linreg = LinearRegression()
logreg = LogisticRegression()
lasso = Lasso(alpha=0.5) # alpha=0 is OLS
ridge = Ridge(alpha=0.5) # alpha=0 is OLS
en = ElasticNet(alpha=0.5, l1_ratio=1) # alpha=0 is OLS, l1_ratio=0 is Ridge
xgbr = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
xgbc = XGBClassifier(n_estimators=100, max_depth=10, eta=0.1, reg_alpha=0, reg_lambda=1, colsample_bylevel=1, colsample_bytree=1, gamma=0)
dtc = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2, min_samples_leaf=1)
rfr = RandomForestRegressor(n_estimators=100, max_depth=10)
rfc = RandomForestClassifier(n_estimators=100, max_depth=10)
gbc = GradientBoostingClassifier(loss='log_loss', learning_rate=.1, n_estimators=100, max_depth=10, sub_sample=.8)
kn = KNeighborsClassifier(n_neighbors=10, weights='uniform', algorithm='auto')

#######################################################################################################################
# simple grid search

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth' : [3, 5, 7]
}

grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5, verbose=2)
grid_search.fit(X, y)

# display best parameters and best score
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# build model with best paremeters
best_rfc = RandomForestClassifier(**grid_search.best_params_).fit(X, y)

#################################################################################################################
"""Model Learning"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state = 2)

# training and predictions
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#################################################################################################################
"""Evaluation"""
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, plot_roc_curve

# regression
r2_score(y_test, y_pred)
mean_squared_error(y_test, y_pred)
mean_absolute_error(y_test, y_pred)

# classification
accuracy_score(y_test, y_pred)
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
roc_auc_score(y_test, y_pred)
f1_score(y_test, y_pred)
plot_roc_curve(model, X_test, y_test)

#################################################################################################################
"""Linear and Logistic regression with statsmodels"""
# linear / logistic regression pull coefficients

import pandas as pd
import statsmodels.api as sm

X = add_constant(X)                        # add intercept
model = sm.OLS(y_train, X_train).fit()     # linear regression
model = sm.Logit(y_train, X_train).fit()   # logistic regression
model.predict(X_test)

print(model.summary())               # show summary
p_vals = pd.DataFrame(model.pvalues) # pull p-values
coefs = pd.DataFrame(model.params)   # pull coefficients

#################################################################################################################
"""Feature Importance"""
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0)
rf.fit(X, y)

feat_importance_dict = {'feature': X.columns,
                        'importance': rf.feature_importances_}

pd.DataFrame(feat_importance_dict)
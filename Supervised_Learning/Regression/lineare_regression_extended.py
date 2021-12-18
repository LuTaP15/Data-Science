"""
This is a Linear Regression model based on the famous wine data set + all required tests for a correctly done LR.

Tests are required to check the following:

The residuals must follow a normal distribution
The residuals are homogeneous, there is homocedasticity
There is no outliers in the errors
There is no autocorrelation in the errors
There is no multicolinearity between the independet variables

This results in the following checks:
1. Normality Check
2. Homocedasticity Check
3. Redidual Outliers Check
4. Redidual Independece Check
5. Multicollinearity Check

"""
# Import libaries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import lilliefors #Kolmogorov-Smirnov
import scipy.stats as scs # QQ plot
from statsmodels.compat import lzip #Homocedascity
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.stats.stattools import durbin_watson
import sklearn.datasets
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Load dataset
data = sklearn.datasets.load_wine()

# Create dataframe
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df.rename(columns={"od280/od315_of_diluted_wines": "test_diluted_wines"}, inplace=True)

# Shape
print(f"Shape of dataframe: {df.shape}")

# Check for zeros
print(f"Zeros in the data: {df.isnull().sum()}")

# Regression Model
model = smf.ols('target ~ '
                'alcohol + ash + alcalinity_of_ash + total_phenols + flavanoids + nonflavanoid_phenols + '
                'color_intensity + hue + test_diluted_wines + proline', data=df).fit()
# Extract the residuals
model_resid = model.resid

##################################################################################
# Test for normality

print("1. Normality Check")
# Kolmogorov-Smirnov test
_, p = lilliefors(model_resid, dist='norm')
print('Kolmogorov-Smirnov:')
print('Not normal | p-value:' if p < 0.05 else 'Normal | p-value:', p)
print('-------------------------------')
# Anderson
stat, p3, _ = scs.anderson(model_resid, dist='norm')
print('Anderson:')
print('Not normal | stat:' if stat > p3[2] else 'Normal | stat:', stat, ':: p-value:', p3[2])

# Histogramm of residuals
model.resid.hist()
plt.show()

# QQPlot
scs.probplot(model_resid, dist='norm', plot=plt)
plt.title('QQ Plot')
plt.show()

##################################################################################
# Ho = Homocedasticity = P > 0.05
# Ha = There's no homocedasticity = p <=0.05
# Homocedasticity test
stat, p, f, fp = sms.het_breuschpagan(model_resid, model.model.exog)
print("2. Homocedasticity Check")
print(f'Test stat: {stat}')
print(f'p-Value: {p}')
print(f'F-Value: {f}')
print(f'f_p_value: {fp}')

# Visualization
plt.scatter(y= model_resid, x=model.predict(), color='red')
plt.hlines(y=0, xmin=0, xmax=4, color='orange')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.show()

##################################################################################
# Residual outliers

outliers = model.outlier_test()

print("3. Residual Outliers Check")
print(outliers.max())
print(outliers.min())

##################################################################################
# Residual independence
print("4. Redidual Independece Check")
print(f'Test stat:\n{durbin_watson(model_resid)}')

##################################################################################
# Multicollinearity
variables = df.drop('target', axis=1)
print("5. Multicollinearity Check")
print(variables.corr())

##################################################################################
# Variation Inflation Factor Test
vif = add_constant(variables)
vif_check = pd.Series([variance_inflation_factor(vif.values, i) for i in range(vif.shape[1])], index=vif.columns)

print(f"Variation Inflation Check:\n{vif_check}")

##################################################################################
# Model Analysis
print(f"Model Analysis:\n{model.summary()}")
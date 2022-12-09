"""
Template for typical Data Transformation tasks
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, Binarizer, OneHotEncoder

# feature indexing based on types
continuous = data.select_dtypes(include=[np.number]).columns # all continuous columns
categorical = data.select_dtypes(exclude=[np.number]).columns # all categorical columns

# generate up to n degree of polynomial terms and interaction terms
data[continuous] = PolynomialFeatures(degree=3, include_bias=False).fit_transform(features)

# standardization and normalization
data[continuous] = StandardScaler().fit_transform(data[continuous])
data[continious] = MinMaxScalar().fit_transform(data[continuous])

# binarization using threshold
data[continuous] = Binarizer(threshold=1).fit_transformer()

# one hot encoding
data[categorical] = OneHotEncoder(drop='first').fit_transform(data[categorical])

# Transformation without sklearn
# normalization, standardization
data = (data-data.mean())/data.std()
data = (data-data.min())/(data.max()-data.min())

# one hot encoding
pd.get_dummies(df, drop_first=True)

# binarization
data[label] = (data[label] == 'category') * 1.0


# Sampling
"""
Alternative Sampling:
Oversampling: SMOTE & ADASYN algorithms
Undersampling: Tomek links and cluster-based
"""
data.sample(1000) # down-sample
data.sample(20000, replace=True) # up-sample
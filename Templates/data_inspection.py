"""
Template for typical Data Inspection tasks with data.
"""

# first 10 rows and last 10 rows for inspection
data.head(10)
data.tail(10)

# column names
data.columns

# column data types
data.dtypes

# type casting if necessary
data['col'] = data['col'].astype(str)
data['col'] = data['col'].astype(int)
data['col'] = data['col'].astype(float)

# num rows, num cols, num null and non-null, variable names
data.info()

# distribution of continuous
data.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])

# data cardinality
data.nunique()

# feature value counts
data['col'].value_counts()

# check null
data.isnull().sum()
data.isnull().mean()

# check or drop duplicates
data[data.duplicated()].shape()

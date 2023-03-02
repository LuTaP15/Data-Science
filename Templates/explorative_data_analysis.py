"""
This serves as a template for explorative data analysis of a new dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

plt.style.use("ggplot")
pd.set_option('display.max_rows', 200)

# Read data
filepath = "./../data/iris_data.csv"
df = pd.read_csv(filepath)

#############################
# Basic understanding of data

# Shape of dataset
print(df.shape)
# Show first 5 rows
print(df.head())
# Show last 5 rows
print(df.tail())
# Display all column names
print(df.columns)
# Display dtype of each row
print(df.dtypes)
# Get a summary
print(df.describe())

#############################
# Data preparation

# Selection columns and copy it to get a completly new dataframe
new_df = df[["'sepal_length', 'sepal_width'"]].copy()

# Dropping columns
df.drop(['sepal_length', 'sepal_width'])

# Convert date columns to datetime format
pd.to_datetime(df["time"])

# Rename columns
df.rename(columns={"old_name": "new_name"})

# Check for NaN
df.isna().sum()

# Check for duplicates
df.duplicated()

# Select all duplicates
df.loc[df.duplicated()]

# Checking data with query
df.query("petal_length == 25.0")


#############################
# Feature understanding

# Univeriate analysis

# Value counts
df["sepal_length"].value_counts()

# Plot the value counts of top 10
ax = df["sepal_length"].value_counts()\
    .head(10)\
    .plot(kind="bar", title="Top 10")

# Plot Distribution of a column
df["sepal_width"].plot(kind="hist",
                       bins=20,
                       title="Sepal width")

# Kernel density estimate plot
df["sepal_width"].plot(kind="kde",
                       bins=20,
                       title="Sepal width")

#############################
# Feature relationships

# Scatterplot
df.plot(kind="scatter",
        x="sepal_width",
        y="petal_width")
plt.show()

# Scatterplot with seaborn
sns.pairplot(df,
             vars=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
             hue="species")
plt.show()

# Heatmap of Korrelation
df_corr = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].dropna().corr()

sns.heatmap(df_corr, annot=True)
"""
Handling missing values in data with misssingno

1. Bar plot to spot missing values
2. Matrix plot to locate the missing values
3. Heatmap for relations between the missing values
"""

# Package imports
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

# Load the Titanic data set
titanic = sns.load_dataset("titanic")

# Gives a bar chart of the missing values
msno.bar(titanic)
plt.show()

# Gives positional information of the missing values
msno.matrix(titanic)
plt.show()

# Gives a heatmap of how missing values are related
msno.heatmap(titanic)
plt.show()
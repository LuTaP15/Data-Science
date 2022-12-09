"""
Template for typical Explorative Data Analysis plotting
"""
import seaborn as sns

# pairwise correlation plots
sns.heatmap(data.corr(), square=True, annot=True)

# visualize nulls
sns.heatmap(data.sample(1000).isnull())

# pairwise comparison (and histogram when row = col)
sns.pairplot(data)

# scatterplot with semi transparency for many dense data points
sns.scatterplot(x='col1',y='col2', data=data, alpha=0.3)

# hexplots
sns.jointplot(x='col1', y='col2', kind='hex')

# boxplots
sns.boxplot(x='col1', y='col2', data=data)

# countplots
sns.countplot(x='col1', y='col2', data=data)

# historgram of features by class in classification
sns.histplot(data=data, x='col', hue='class')

# density plot (similar to histogram but smoother)
data.plot.density()
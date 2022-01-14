import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
#import pybaobabdt as pbb
import matplotlib.pyplot as plt

df = pd.read_csv("./../../data/iris_data.csv")

print(df.head())

y = list(df['species'])
features = list(df.columns)
target = df['species']
features.remove('species')
X = df.loc[:, features]

clf = DecisionTreeClassifier().fit(X, y)

##########################################################################################
fig = plt.figure(figsize=(20, 8))
_ = plot_tree(
    clf,
    feature_names=features,
    class_names=list(set(target)),
    filled=True,
    fontsize=11,
    label='all',
    rounded=True
)
plt.show()
##########################################################################################
# Needs pygraphviz

#ax = pbb.drawTree(clf, size=10, dpi=300, features=features, ratio=0.8, colormap='Viridis')
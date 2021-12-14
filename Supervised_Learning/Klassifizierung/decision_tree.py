import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pybaobabdt as pbb

df = pd.read_csv("./../../data/iris_data.csv")

print(df.head())

y = list(df['species'])
features = list(df.columns)
target = df['species']
features.remove('species')
X = df.loc[:, features]

clf = DecisionTreeClassifier().fit(X, y)

ax = pbb.drawTree(clf, size=10, dpi=300, features=features, ratio=0.8, colormap='Viridis')
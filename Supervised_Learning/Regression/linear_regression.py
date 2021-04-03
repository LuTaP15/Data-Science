"""
Beispiel Implementierung einer LinearRegression zur Vorhersage des Preises eines Gebrauchtswagens.
Datensatz besteht aus Einträge für Gebrauchtwagen von Ebay.

"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("./../../data/autos_prepared.csv")
print(df.head())

X = df[["kilometer"]]
Y = df[["price"]]

# Explore
#plt.scatter(X,Y)
#plt.show()


# Lineare Regression
model = LinearRegression()
model.fit(X, Y)

print("Intercept: " + str(model.intercept_))
print("Coef: " + str(model.coef_))
print(f"Gleichung: {str(model.coef_)} * x + {str(model.intercept_)}")

print(f"Der Preis für ein Auto mit 50.000km liegt bei etwa: {int(model.predict([[50000]]))}")


#############################################
min_x = min(df["kilometer"])
max_x = max(df["kilometer"])

predicted = model.predict([[min_x], [max_x]])

plt.scatter(df["kilometer"], df["price"])
plt.plot([min_x, max_x], predicted, color='red')
plt.show()
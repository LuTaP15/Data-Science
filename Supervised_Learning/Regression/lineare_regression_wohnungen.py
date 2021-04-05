"""
Beispiel Implementierung einer LinearRegression zur Vorhersage des Verkaufspreises pro Quadratmeter.

"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv("./../../data/wohnungspreise.csv")

# Explore the data
#plt.scatter(df["Quadratmeter"], df["Verkaufspreis"])
#plt.show()

X = df[["Quadratmeter"]]
Y = df[["Verkaufspreis"]]

model = LinearRegression()
model.fit(X, Y)

print("Intercept: " + str(model.intercept_))
print("Coef: " + str(model.coef_))
print(f"Gleichung: {str(model.coef_)} * x + {str(model.intercept_)}")

print(model.predict([[40], [80]]))

min_x = min(df["Quadratmeter"])
max_x = max(df["Quadratmeter"])

predicted = model.predict([[min_x], [max_x]])


plt.scatter(df["Quadratmeter"], df["Verkaufspreis"])
plt.plot([min_x, max_x], predicted, color='red')
plt.show()
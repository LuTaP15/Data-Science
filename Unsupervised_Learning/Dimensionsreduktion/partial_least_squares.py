"""
Applying Partial Least Squares for dimension reduction on the breast cancer dataset with 30 features:

- Load and split the breast cancer dataset into training and test sets and standardize the data
- Use of PLS to reduce dimensions from 30 to 3 features
- Trained linear regression model on the transformed training data
- Calculates the mean squared error (MSE) between the predicted values and the true test labels

"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the diabetes dataset
X, y = load_breast_cancer(return_X_y=True)

# Split the data into training and test sets and standardize it
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit a PLS model to the training data and transform the training and test data
pls = PLSRegression(n_components=3)
pls.fit(X_train, y_train)
X_train_transformed = pls.transform(X_train)
X_test_transformed = pls.transform(X_test)

# Fit a linear regression model to the transformed training data
lr = LinearRegression()
lr.fit(X_train_transformed, y_train)

# Predict the response variable on the transformed test data
y_pred = lr.predict(X_test_transformed)

# Calculate the mean squared error of the model's predictions
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

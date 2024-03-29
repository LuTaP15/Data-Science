"""
Project to predict the price of californian houses

Pipeline and Gridsearch for RandomForestRegression, XGBoost

#######################
1. EDA
2. Random Forest Model
3. XGBoost Model
4. Evaluation
#######################
Settings:
MODEL_SELECT = "XGBOOST" # Options "RFR" or "XGBOOST"
VIEW_ALL_DATA = True
DISPLAY_PRECISION_2 = False
PLOTTING_ON = False
EXPLORE_DATA_ON = False
EVALUATION_ON = False
"""

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

##########################################################################################
# Constants
MODEL_SELECT = "RFR"                # Options "RFR" or "XGBOOST"
VIEW_ALL_DATA = True
DISPLAY_PRECISION_2 = False
PLOTTING_ON = False
EXPLORE_DATA_ON = False
EVALUATION_ON = False

##########################################################################################
# Functions


def load_data(fetch_california_housing):
    # Function to load data from sklearn
    data = pd.DataFrame(fetch_california_housing.data, columns=fetch_california_housing.feature_names)
    data['AveHouseVal'] = fetch_california_housing.target*100000
    return data


def plot_data_on_map(df):
    # Function to plot data on map
    plt.figure(figsize=(10, 8))
    plt.scatter(df['Latitude'], df['Longitude'], c=df['AveHouseVal'], s=df['Population']/100)
    plt.colorbar()
    plt.show()
    
def plot_histograms(df):
    # Function to plot the data as histograms
    df.hist(bins=50, figsize=(20, 15))
    plt.show()


def explore_data(df):
    # Function to explore the data
    # Check features and measurements
    print(f"Amount of features: {df.shape[0]}, Amount of measurements: {df.shape[1]}")
    # Print feature names
    print(list(df.columns))
    if VIEW_ALL_DATA:
        pd.set_option("display.max.columns", None)
    if DISPLAY_PRECISION_2:
        pd.set_option("display.precision", 2)

    # Check first 5 entries
    print("First 5 entries in the data set.")
    print(df.head())
    # Check last 5 entries
    print("Last 5 entries in the data set.")
    print(df.tail())

    # Check datatypes of features
    print(df.info())

    # Statistical analysis
    print(df.describe())

    # Check for NANs
    check_nan = df.isnull().values.any()
    print(check_nan)

    # Correlation of housing price with other features
    print(correlation_to_house_value(df))


def correlation_to_house_value(df):
    corr_matrix = df.corr()
    return corr_matrix["AveHouseVal"].sort_values(ascending=False)


def cal_smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def random_forest_regression(X_train, y_train, X_test):
    # Random forest model
    rfr = RandomForestRegressor(n_estimators=100, max_depth=10)
    # Train model
    rfr.fit(X_train, y_train)
    # Predict
    return rfr.predict(X_test)


def xgboost_regression(X_train, y_train, X_test):
    # XGBoost model
    xgbr = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
    # Train model
    xgbr.fit(X_train, y_train)
    # Predict
    return xgbr.predict(X_test)


def evaluation(y_test, y_pred):
    # Function to calculate metrics
    # R^2
    r2 = r2_score(y_test, y_pred)
    # RMSE
    rmse = mean_squared_error(y_test, y_pred)
    # MAE
    mae = mean_absolute_error(y_test, y_pred)
    # MAPE
    mape = mean_absolute_percentage_error(y_test, y_pred)
    # SMAPE
    smape = cal_smape(y_test, y_pred)
    return r2, rmse, mae, mape, smape


if __name__ == "__main__":
    # Load data
    fetch_california_housing = fetch_california_housing()
    df = load_data(fetch_california_housing)
    # Explore data
    if EXPLORE_DATA_ON:
        explore_data(df)
    if PLOTTING_ON:
        # Plot Histograms
        plot_histograms(df)
        # Plot data Population and housing price
        plot_data_on_map(df)

    # Data Preprocessing
    # Selecting Features and Label
    y = df.pop("AveHouseVal")
    X = df
    # Split data in train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)

    # Skalieren der Daten
    X_train, X_test = scale_data(X_train, X_test)

    if MODEL_SELECT == "RFR":
        # Random Forest Regression
        print("Selected model: Random Forest Regression")
        y_pred = random_forest_regression(X_train, y_train, X_test)
        EVALUATION_ON = True

    elif MODEL_SELECT == "XGBOOST":
        # Xgboost Regression
        print("Selected model: XGBOOST Regression")
        y_pred = xgboost_regression(X_train, y_train, X_test)
        EVALUATION_ON = True

    else:
        print("Select a model!")

    # Evaluation
    if EVALUATION_ON:
        r2, rmse, mae, mape, smape = evaluation(y_test, y_pred)
        print(f"Root Squared: {r2}")
        print(f"Root Mean Square Error: {rmse}")
        print(f"Mean Absolut Error: {mae}")
        print(f"Mean Absolute Percentage Error: {mape}")
        print(f"Symmetric Mean Absolute Percentage Error: {smape}")

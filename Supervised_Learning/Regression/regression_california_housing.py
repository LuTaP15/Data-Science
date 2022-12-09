"""
Project to predict the price of californian houses

Pipeline and Gridsearch for RandomForestRegression, XGBoost

#######################
1. EDA
2. Random Forest Model
3. XGBoost Model
4. Evaluation

"""

# Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

##########################################################################################
# Constants
MODEL_SELECT = "XGBOOST"                # Options "RFR" or "XGBOOST"
VIEW_ALL_DATA = True
DISPLAY_PRECISION_2 = False
PLOTTING_ON = False
EXLORE_DATA_ON = False

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


def evaluation(y_test, y_pred):
    # Function to calculate metrics
    # R^2
    r2 = r2_score(y_test, y_pred)
    # RMSE
    rmse = mean_squared_error(y_test, y_pred)
    # MAE
    mae = mean_absolute_error(y_test, y_pred)
    # MAPE
    # SMAPE
    return r2, rmse, mae


if __name__ == "__main__":
    # Load data
    fetch_california_housing = fetch_california_housing()
    df = load_data(fetch_california_housing)
    # Explore data
    if EXLORE_DATA_ON:
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

    if MODEL_SELECT == "RFR":
        # Random forest model
        rfr = RandomForestRegressor(n_estimators=100, max_depth=10)
        # Train model
        rfr.fit(X_train, y_train)
        # Predict
        y_pred = rfr.predict(X_test)

    elif MODEL_SELECT == "XGBOOST":
        # XGBoost model
        xgbr = XGBRegressor(n_estimators=1000, max_depth=7, eta=0.1, subsample=0.7, colsample_bytree=0.8)
        # Train model
        xgbr.fit(X_train, y_train)
        # Predict
        y_pred = xgbr.predict(X_test)

    else:
        print("Select a model!")

    # Evaluation
    r2, rmse, mae = evaluation(y_test, y_pred)
    print(f"R2: {r2}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
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

##########################################################################################
# Constants
VIEW_ALL_DATA = True
DISPLAY_PRECISION_2 = False

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


def explore_data(df):
    # Function to explore the data
    # Check features and measurements
    print(f"Amount of features: {df.shape[0]}, Amount of measurements: {df.shape[1]}")
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

    # Print Correlationmatrix


def evaluation():
    # Function to calculate metrics
    print("eval")
    # R^2
    # RMSE
    # MAE
    # MAPE
    # SMAPE


if __name__ == "__main__":
    # Load data
    fetch_california_housing = fetch_california_housing()
    df = load_data(fetch_california_housing)
    # Explore data
    explore_data(df)
    # Random forest model
    # XGBoost model
    # Evaluation

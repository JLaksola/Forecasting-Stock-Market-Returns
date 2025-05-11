import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
file_path = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/processed/merged_data.csv"
df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

# Let's convert the 10Y stock returns to a percentage
df["10Y_stock_returns"] = df["10Y_stock_returns"] * 100

# Define the inverse CAPE variable
df["inv_CAPE"] = 1 / df["CAPE"]

# Let's initate the multiple feature linear regression model
# Define the features
features = [
    "CPI",
    "inv_CAPE",
    "SVAR",
    "Real_10Y_Yield",
    "BondVol",
    "DFR",
]
# Define the target variable
target = "10Y_stock_returns"
# Initialize rolling train and test sets
train_start = "1927-05-01"
test_start = "1960-01-01"
test_end = "2015-02-01"

# Initialize empty RMSE list
rmse_list = []
predictions = []  # Store predicted values
actuals = []  # Store actual values
dates = []  # Store corresponding dates

# Rolling Forecast Loop
for date in pd.date_range(test_start, test_end, freq="MS"):  # Monthly steps
    train_end = (pd.Timestamp(date) - pd.DateOffset(years=10, months=1)).strftime(
        "%Y-%m-%d"
    )
    train = df.loc[train_start:train_end]  # Only use past data (10 years before)
    test_sample = df.loc[date:date]  # Predict one step ahead

    # Train model
    X_train = sm.add_constant(train[features])
    y_train = train[target]
    model = sm.OLS(y_train, X_train).fit()

    # Prepare test sample
    X_test = sm.add_constant(test_sample[features], has_constant="add")
    y_test = test_sample[target]

    # Predict & Calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error([y_test], y_pred))
    rmse_list.append(rmse)
    predictions.append(float(y_pred.values[0]))
    actuals.append(y_test.values[0])
    dates.append(date)

# Convert results to DataFrame
results_df = pd.DataFrame(
    {
        "Date": dates,
        "Predicted": [float(p) for p in predictions],  # ensure clean floats
        "Actual": actuals,
        "RMSE": rmse_list,
    }
)
results_df.set_index("Date", inplace=True)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Predicted"], label="Predicted", color="blue")
plt.plot(results_df.index, results_df["Actual"], label="Actual", color="red")
plt.title("Predicted vs Actual Returns")
plt.xlabel("Date")
plt.ylabel("Returns")
plt.legend()
plt.grid()
plt.show()


# To CSV
results_df.to_csv(
    "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/reports/results_linear_multi.csv",
    index=True,
)


# Let's initiate the inverse CAPE based linear regression model
train_start = "1927-05-01"
test_start = "1960-01-01"
test_end = "2015-02-01"

# Initialize empty RMSE list
rmse_list = []
predictions = []  # Store predicted values
actuals = []  # Store actual values
dates = []  # Store corresponding dates


# Rolling Forecast Loop
for date in pd.date_range(test_start, test_end, freq="MS"):  # Monthly steps
    train_end = (pd.Timestamp(date) - pd.DateOffset(years=10, months=1)).strftime(
        "%Y-%m-%d"
    )
    train = df.loc[train_start:train_end]  # Only use past data (10 years before)
    test_sample = df.loc[date:date]  # Predict one step ahead

    # Train model
    X_train = sm.add_constant(train["inv_CAPE"], has_constant="add")
    y_train = train[target]
    model = sm.OLS(y_train, X_train).fit()

    # Prepare test sample
    X_test = sm.add_constant(test_sample["inv_CAPE"], has_constant="add")
    y_test = test_sample[target]

    # Predict & Calculate RMSE
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error([y_test], y_pred))
    rmse_list.append(rmse)
    predictions.append(float(y_pred.values[0]))
    actuals.append(y_test.values[0])
    dates.append(date)

# Convert results to DataFrame
results_df = pd.DataFrame(
    {
        "Date": dates,
        "Predicted": [float(p) for p in predictions],  # ensure clean floats
        "Actual": actuals,
        "RMSE": rmse_list,
    }
)
results_df.set_index("Date", inplace=True)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Predicted"], label="Predicted", color="blue")
plt.plot(results_df.index, results_df["Actual"], label="Actual", color="red")
plt.title("Predicted vs Actual Returns (Inverse CAPE)")
plt.xlabel("Date")
plt.ylabel("Returns")
plt.legend()
plt.grid()
plt.show()

# To CSV
results_df.to_csv(
    "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/reports/linear_inv_CAPE.csv",
    index=True,
)

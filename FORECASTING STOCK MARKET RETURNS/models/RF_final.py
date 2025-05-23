import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
file_path = "/Users/kayttaja/Desktop/bachelor_proj/data/processed/merged_data.csv"
df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

# Select Features and Target
features = [
    "CPI",
    "CAPE",
    "Real_10Y_Yield",
    "ReturnVol",
    "BondVol",
    "SVAR",
    "DFY",
    "DFR",
    "TBL",
]
target = "10Y_stock_returns"

# Convert target to percentage
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Define Rolling Forecast Timeframe
TRAIN_START = "1927-05-01"
TEST_START = "1960-01-01"
TEST_END = "2015-02-01"

# Initialize storage lists
dates = []
predictions = []
actuals = []
rmse_list = []

# Define Hyperparameter Grid
param_grid = {
    "n_estimators": [50, 100],  # Number of trees
    "max_depth": [5, 10, None],  # Depth of each tree
    "min_samples_split": [2, 5],  # Min samples to split a node
    "min_samples_leaf": [1, 2],  # Min samples per leaf
}


# Rolling Forecast Loop
date_range = pd.date_range(TEST_START, TEST_END, freq="MS")  # Monthly steps

for date in date_range:
    # Define Rolling Window (10-year past data)
    train_end = (date - pd.DateOffset(years=10)).strftime("%Y-%m-%d")

    # Get Training and Testing Data
    train_data = df.loc[TRAIN_START:train_end]
    test_data = df.loc[date:date]  # Predict one month ahead

    # Split Features and Target
    X_train = train_data[features]
    y_train = train_data[target]

    # GridSearchCV (Hyperparameter Tuning)
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,  # Cross-validation within rolling window
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    # Retrieve best model
    best_rf = grid_search.best_estimator_

    # Test Data
    X_test = test_data[features]
    y_test = test_data[target]

    # Predict next step
    y_pred = best_rf.predict(X_test)[0]

    # Compute RMSE
    rmse_step = np.sqrt(mean_squared_error([y_test], [y_pred]))

    # Store results
    dates.append(date)
    actuals.append(y_test.values[0])
    predictions.append(y_pred)
    rmse_list.append(rmse_step)


# Save the Results to a DataFrame
results_df = pd.DataFrame(
    {"Date": dates, "Actual": actuals, "Predicted": predictions, "RMSE": rmse_list}
)
results_df.set_index("Date", inplace=True)
# Compute average RMSE
print(results_df["RMSE"].mean())

# Plot Predictions vs Actuals
plt.figure(figsize=(12, 6))
plt.plot(results_df["Date"], results_df["Actual"], label="Actual", color="blue")
plt.plot(
    results_df["Date"],
    results_df["Predicted"],
    label="Predicted (RF)",
    color="red",
    linestyle="--",
)
plt.title("Random Forest Rolling Forecast (Hyperparameter Tuned)")
plt.xlabel("Date")
plt.ylabel("10-Year Annualized Return")
plt.legend()
plt.grid(True)
plt.show()


# This is for google colab since I don't have the compute capability to run the code on my local machine
from cuml.ensemble import RandomForestRegressor as cuRF
from sklearn.model_selection import TimeSeriesSplit

# Load the data
file_path = "/content/drive/MyDrive/Colab_Notebooks/merged_data.csv"
df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

# Select Features and Target
features = [
    "CPI",
    "CAPE",
    "Real_10Y_Yield",
    "ReturnVol",
    "BondVol",
    "SVAR",
    "DFY",
    "DFR",
    "TBL",
]
target = "10Y_stock_returns"

# Convert target to percentage
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])


# Define Rolling Forecast Timeframe
TRAIN_START = "1927-05-01"
TEST_START = "1960-01-01"
TEST_END = "2015-02-01"

# Initialize storage lists
dates = []
predictions = []
actuals = []
rmse_list = []

# Define Hyperparameter Grid
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]}

# Rolling Forecast Loop
date_range = pd.date_range(TEST_START, TEST_END, freq="MS")

for date in date_range:
    # Define Rolling Window (10-year past data)
    train_end = (date - pd.DateOffset(years=10, months=1)).strftime("%Y-%m-%d")

    # Get Training and Testing Data
    train_data = df.loc[TRAIN_START:train_end]
    test_data = df.loc[date:date]

    # Split Features and Target
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    # **Perform Hyperparameter Tuning using TimeSeriesSplit**
    tscv = TimeSeriesSplit(n_splits=5)  # Rolling cross-validation

    best_score = float("inf")
    best_params = {}

    for n_estimators in param_grid["n_estimators"]:
        for max_depth in param_grid["max_depth"]:
            mse_cv = []  # Store validation scores for each fold

            for train_idx, val_idx in tscv.split(X_train):
                X_train_fold, X_val_fold = (
                    X_train.iloc[train_idx],
                    X_train.iloc[val_idx],
                )
                y_train_fold, y_val_fold = (
                    y_train.iloc[train_idx],
                    y_train.iloc[val_idx],
                )

                rf_gpu = cuRF(n_estimators=n_estimators, max_depth=max_depth)
                rf_gpu.fit(X_train_fold, y_train_fold)
                y_val_pred = rf_gpu.predict(X_val_fold)

                mse_fold = np.mean((y_val_fold - y_val_pred) ** 2)
                mse_cv.append(mse_fold)

            avg_mse = np.mean(mse_cv)  # Average validation error

            if avg_mse < best_score:
                best_score = avg_mse
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

    # **Train Final Model with Best Hyperparameters on Full Training Data**
    best_rf = cuRF(**best_params)
    best_rf.fit(X_train, y_train)

    # **Test Final Model on Unseen Test Data**
    y_pred = best_rf.predict(X_test)[0]

    # Compute RMSE
    rmse_step = np.sqrt(mean_squared_error([y_test], [y_pred]))

    # Store results
    dates.append(date)
    actuals.append(y_test.values[0])
    predictions.append(y_pred)
    rmse_list.append(rmse_step)

    print(f"Date: {date}, Best Params: {best_params}, RMSE: {rmse_step}")

# Load the data
file_path = "/Users/kayttaja/Desktop/bachelor_proj/reports/results_rf.csv"
results_df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
# Compute average RMSE
print(results_df["RMSE"].mean() * 100)

# Plot Predictions vs Actuals
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Actual"], label="Actual", color="blue")
plt.plot(
    results_df.index,
    results_df["Prediction"],
    label="Predicted (RF)",
    color="red",
    linestyle="--",
)
plt.title("Random Forest Rolling Forecast (Hyperparameter Tuned)")
plt.xlabel("Date")
plt.ylabel("10-Year Annualized Return")
plt.legend()
plt.grid(True)
plt.show()


# Let's use only CAPE
# Select Features and Target
features = ["CAPE"]
target = "10Y_stock_returns"

# Convert target to percentage
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])


# Define Rolling Forecast Timeframe
TRAIN_START = "1927-05-01"
TEST_START = "1960-01-01"
TEST_END = "2015-02-01"

# Initialize storage lists
dates = []
predictions = []
actuals = []
rmse_list = []

# Define Hyperparameter Grid
param_grid = {"n_estimators": [50, 100, 200], "max_depth": [5, 10, 20]}

# Rolling Forecast Loop
date_range = pd.date_range(TEST_START, TEST_END, freq="MS")

for date in date_range:
    # Define Rolling Window (10-year past data)
    train_end = (date - pd.DateOffset(years=10, months=1)).strftime("%Y-%m-%d")

    # Get Training and Testing Data
    train_data = df.loc[TRAIN_START:train_end]
    test_data = df.loc[date:date]

    # Split Features and Target
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    # **Perform Hyperparameter Tuning using TimeSeriesSplit**
    tscv = TimeSeriesSplit(n_splits=5)  # Rolling cross-validation

    best_score = float("inf")
    best_params = {}

    for n_estimators in param_grid["n_estimators"]:
        for max_depth in param_grid["max_depth"]:
            mse_cv = []  # Store validation scores for each fold

            for train_idx, val_idx in tscv.split(X_train):
                X_train_fold, X_val_fold = (
                    X_train.iloc[train_idx],
                    X_train.iloc[val_idx],
                )
                y_train_fold, y_val_fold = (
                    y_train.iloc[train_idx],
                    y_train.iloc[val_idx],
                )

                rf_gpu = cuRF(n_estimators=n_estimators, max_depth=max_depth)
                rf_gpu.fit(X_train_fold, y_train_fold)
                y_val_pred = rf_gpu.predict(X_val_fold)

                mse_fold = np.mean((y_val_fold - y_val_pred) ** 2)
                mse_cv.append(mse_fold)

            avg_mse = np.mean(mse_cv)  # Average validation error

            if avg_mse < best_score:
                best_score = avg_mse
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

    # **Train Final Model with Best Hyperparameters on Full Training Data**
    best_rf = cuRF(**best_params)
    best_rf.fit(X_train, y_train)

    # **Test Final Model on Unseen Test Data**
    y_pred = best_rf.predict(X_test)[0]

    # Compute RMSE
    rmse_step = np.sqrt(mean_squared_error([y_test], [y_pred]))

    # Store results
    dates.append(date)
    actuals.append(y_test.values[0])
    predictions.append(y_pred)
    rmse_list.append(rmse_step)

    print(f"Date: {date}, Best Params: {best_params}, RMSE: {rmse_step}")

# Convert results to DataFrame
results_df = pd.DataFrame(
    {"Date": dates, "Actual": actuals, "Prediction": predictions, "RMSE": rmse_list}
)
results_df = pd.DataFrame(
    {"Date": dates, "Actual": actuals, "Prediction": predictions, "RMSE": rmse_list}
)
results_df.set_index(
    "Date", inplace=True
)  # Setting 'Date' as index immediately after creation

# Load the data
file_path = "/Users/kayttaja/Desktop/bachelor_proj/reports/results_rf_CAPE.csv"
results_df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
# Compute average RMSE
print(results_df["RMSE"].mean() * 100)
# Plot Predictions vs Actuals
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Actual"], label="Actual", color="blue")
plt.plot(
    results_df.index,
    results_df["Prediction"],
    label="Predicted (RF)",
    color="red",
    linestyle="--",
)
plt.title("Random Forest Rolling Forecast (Hyperparameter Tuned)")
plt.xlabel("Date")
plt.ylabel("10-Year Annualized Return")
plt.legend()
plt.grid(True)
plt.show()


# CAPE, DFR, DFY
# Select Features and Target
features = ["CAPE", "DFR", "DFY"]
target = "10Y_stock_returns"

# Convert target to percentage
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])


# Define Rolling Forecast Timeframe
TRAIN_START = "1927-05-01"
TEST_START = "1960-01-01"
TEST_END = "2015-02-01"

# Initialize storage lists
dates = []
predictions = []
actuals = []
rmse_list = []

# Define Hyperparameter Grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20],
}

# Rolling Forecast Loop
date_range = pd.date_range(TEST_START, TEST_END, freq="MS")

for date in date_range:
    # Define Rolling Window (10-year past data)
    train_end = (date - pd.DateOffset(years=10, months=1)).strftime("%Y-%m-%d")

    # Get Training and Testing Data
    train_data = df.loc[TRAIN_START:train_end]
    test_data = df.loc[date:date]

    # Split Features and Target
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]

    # **Perform Hyperparameter Tuning using TimeSeriesSplit**
    tscv = TimeSeriesSplit(n_splits=5)  # Rolling cross-validation

    best_score = float("inf")
    best_params = {}

    for n_estimators in param_grid["n_estimators"]:
        for max_depth in param_grid["max_depth"]:
            mse_cv = []  # Store validation scores for each fold

            for train_idx, val_idx in tscv.split(X_train):
                X_train_fold, X_val_fold = (
                    X_train.iloc[train_idx],
                    X_train.iloc[val_idx],
                )
                y_train_fold, y_val_fold = (
                    y_train.iloc[train_idx],
                    y_train.iloc[val_idx],
                )

                rf_gpu = cuRF(n_estimators=n_estimators, max_depth=max_depth)
                rf_gpu.fit(X_train_fold, y_train_fold)
                y_val_pred = rf_gpu.predict(X_val_fold)

                mse_fold = np.mean((y_val_fold - y_val_pred) ** 2)
                mse_cv.append(mse_fold)

            avg_mse = np.mean(mse_cv)  # Average validation error

            if avg_mse < best_score:
                best_score = avg_mse
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

    # **Train Final Model with Best Hyperparameters on Full Training Data**
    best_rf = cuRF(**best_params)
    best_rf.fit(X_train, y_train)

    # **Test Final Model on Unseen Test Data**
    y_pred = best_rf.predict(X_test)[0]

    # Compute RMSE
    rmse_step = np.sqrt(mean_squared_error([y_test], [y_pred]))

    # Store results
    dates.append(date)
    actuals.append(y_test.values[0])
    predictions.append(y_pred)
    rmse_list.append(rmse_step)

    print(f"Date: {date}, Best Params: {best_params}, RMSE: {rmse_step}")

# Convert results to DataFrame
results_df = pd.DataFrame(
    {"Date": dates, "Actual": actuals, "Prediction": predictions, "RMSE": rmse_list}
)
results_df = pd.DataFrame(
    {"Date": dates, "Actual": actuals, "Prediction": predictions, "RMSE": rmse_list}
)
results_df.set_index(
    "Date", inplace=True
)  # Setting 'Date' as index immediately after creation


# Load the data
file_path = "/Users/kayttaja/Desktop/Bachelor_proj/reports/results_rf_CAPE_DFR_DFY.csv"
results_df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
# Compute average RMSE
print(results_df["RMSE"].mean() * 100)
# Plot Predictions vs Actuals
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Actual"], label="Actual", color="blue")
plt.plot(
    results_df.index,
    results_df["Prediction"],
    label="Predicted (RF)",
    color="red",
    linestyle="--",
)
plt.title("Random Forest Rolling Forecast (Hyperparameter Tuned)")
plt.xlabel("Date")
plt.ylabel("10-Year Annualized Return")
plt.legend()
plt.grid(True)
plt.show()


# DFR and CAPE
# Load the data
file_path = "/Users/kayttaja/Desktop/Bachelor_proj/reports/results_rf_CAPE_DFR.csv"
results_df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
# Compute average RMSE
print(results_df["RMSE"].mean() * 100)
# Plot Predictions vs Actuals
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Actual"], label="Actual", color="blue")
plt.plot(
    results_df.index,
    results_df["Prediction"],
    label="Predicted (RF)",
    color="red",
    linestyle="--",
)
plt.title("Random Forest Rolling Forecast (Hyperparameter Tuned)")
plt.xlabel("Date")
plt.ylabel("10-Year Annualized Return")
plt.legend()
plt.grid(True)
plt.show()

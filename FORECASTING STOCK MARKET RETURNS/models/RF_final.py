import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit


# Load the data
file_path = r"C:\Users\Käyttäjä\Desktop\Bachelor_proj\data\processed\merged_data.csv"
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

# Load the data
file_path = r"C:\Users\Käyttäjä\Desktop\Bachelor_proj\data\processed\merged_data.csv"
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
file_path = r"C:\Users\Käyttäjä\Desktop\Bachelor_proj\reports\results_rf_CAPE.csv"
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



# Let's build the RF model using my CV methodology
# Load the data
file_path = r"C:\Users\Käyttäjä\Desktop\Bachelor_proj\data\processed\merged_data.csv"
df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

# A novel CV measure
# Custom time series split that uses rmse for evaluation
def rolling_origin_cv_rmse(
    X, y, model, horizon=120, min_train_size=60, step=1, val_window=12
):
    """
    Perform rolling-origin cross-validation and compute RMSE for each fold.

    Parameters:
        X (array-like): Feature matrix (2D).
        y (array-like): Target array (1D).
        model: A scikit-learn style model with .fit() and .predict().
        horizon (int): Forecast horizon (e.g., 120 for 10 years).
        min_train_size (int): Minimum number of samples to start training.
        step (int): Step size to move the rolling window.
        verbose (bool): If True, print RMSE per fold.

    Returns:
        rmse_list (list): List of RMSE values per fold.
        y_preds (list): List of predicted values.
        y_vals (list): List of actual values.
    """

    rmse_list = []
    mae_list = []
    y_preds = []
    y_vals = []

    n_samples = len(X)

    for t in range(min_train_size, n_samples - val_window - horizon + 1, step):
        X_train = X[:t]
        y_train = y[:t]

        X_val = X[t + horizon : t + horizon + val_window]
        y_val = y[t + horizon : t + horizon + val_window]

        if len(y_val) < val_window:
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        errors = y_val - y_pred
        rmse = np.sqrt(np.mean(errors**2))
        mae = np.mean(np.abs(errors))

        rmse_list.append(rmse)
        mae_list.append(mae)
        y_preds.extend(y_pred)
        y_vals.extend(y_val)

    return rmse_list, mae_list, y_preds, y_vals


# Custom grid search cv
def custom_grid_search_cv(X, y, model_class, param_grid, cv_func, **cv_kwargs):
    from itertools import product

    param_names = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))

    best_score = float("inf")
    best_params = None
    results = []

    for combo in param_combinations:
        params = dict(zip(param_names, combo))
        model = model_class(**params)

        rmse_list, *_ = cv_func(X, y, model, **cv_kwargs)
        mean_rmse = np.mean(rmse_list)

        results.append({**params, "mean_rmse": mean_rmse})

        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params

    return best_params, best_score, results

df = df.copy()
df["10Y_stock_returns"] = df["10Y_stock_returns"] * 100

# Let's use only CAPE
# --- Config ---
features = ["CAPE"]
target = "10Y_stock_returns"
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20]
}
TRAIN_START = "1927-05-01"
TEST_START = "1960-01-01"
TEST_END = "2015-02-01"

# --- Preprocessing ---
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# --- Storage ---
dates = []
predictions = []
actuals = []
rmse_list = []
best_params_list = []

# --- Rolling Forecast ---
date_range = pd.date_range(TEST_START, TEST_END, freq="MS")

for date in date_range:
    train_end = (date - pd.DateOffset(years=10, months=1)).strftime("%Y-%m-%d")
    
    train_data = df.loc[TRAIN_START:train_end]
    test_data = df.loc[date:date]

    X_train = train_data[features].values
    y_train = train_data[target].values
    X_test = test_data[features].values
    y_test = test_data[target].values

    # Custom Grid Search with Your Expanding CV
    best_params, best_score, _ = custom_grid_search_cv(
        X_train, y_train,
        model_class=RandomForestRegressor,
        param_grid=param_grid,
        cv_func=rolling_origin_cv_rmse,
        horizon=120,
        min_train_size=120,
        step=12,
        val_window=12
    )

    # Train best model on full training set
    best_rf = RandomForestRegressor(**best_params, random_state=42)
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)[0]

    # Evaluate
    rmse_step = np.sqrt(mean_squared_error([y_test[0]], [y_pred]))

    # Store results
    best_params_list.append(best_params)
    dates.append(date)
    predictions.append(y_pred)
    actuals.append(y_test[0])
    rmse_list.append(rmse_step)

    print(f"Date: {date.date()}, Best Params: {best_params}, RMSE: {rmse_step:.4f}")

# Save the results
results_rf_CAPE = pd.DataFrame({
    "Date": dates,
    "Prediction": predictions,
    "Actual": actuals,
    "RMSE": rmse_list,
    "Best_Params": best_params_list
}).set_index("Date")

results_rf_CAPE.to_csv("results_rf_CAPE.csv")

results_df = pd.read_csv(r"C:\Users\Käyttäjä\Desktop\Bachelor_proj\models\results_rf_CAPE.csv", index_col="Date", parse_dates=True)

print(results_df["RMSE"].mean())

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


# I have created the inverse of CAPE and a new feature subset
# Let's make a RF model with these new feature subsets
# First the multi feature model not using the novel CV
# Select Features and Target
df["inv_CAPE"] = 1 / df["CAPE"]
df["10Y_stock_returns"] = df["10Y_stock_returns"] * 100
features = ["CPI", "inv_CAPE", "SVAR", "Real_10Y_Yield", "BondVol", "DFR"]
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


    # Perform Hyperparameter Tuning using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    best_score = float("inf")
    best_params = {}

    for n_estimators in param_grid["n_estimators"]:
        for max_depth in param_grid["max_depth"]:
            mse_cv = []  # Store validation scores for each fold

            for train_idx, val_idx in tscv.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1  # Leverage parallelism
                )
                rf.fit(X_train_fold, y_train_fold)
                y_val_pred = rf.predict(X_val_fold)

                mse_fold = mean_squared_error(y_val_fold, y_val_pred)
                mse_cv.append(mse_fold)

            avg_mse = np.mean(mse_cv)

            if avg_mse < best_score:
                best_score = avg_mse
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

    # Train Final Model with Best Hyperparameters on Full Training Data
    best_rf = RandomForestRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    best_rf.fit(X_train, y_train)

    # Test Final Model on Unseen Test Data
    y_pred = best_rf.predict(X_test)

    # Compute RMSE
    rmse_step = np.sqrt(mean_squared_error(y_test, y_pred))

    # Store results
    dates.append(date)
    actuals.append(y_test.values[0])
    predictions.append(y_pred[0])
    rmse_list.append(rmse_step)

    print(f"Date: {date.strftime('%Y-%m-%d')}, Best Params: {best_params}, RMSE: {rmse_step:.4f}")

# Save the results
results_rf_CAPE = pd.DataFrame({
    "Date": dates,
    "Prediction": predictions,
    "Actual": actuals,
    "RMSE": rmse_list,
}).set_index("Date")

results_rf_CAPE.to_csv("results_rf_CAPE.csv")


print(results_rf_CAPE["RMSE"].mean())

# Plot Predictions vs Actuals
plt.figure(figsize=(12, 6))
plt.plot(results_rf_CAPE.index, results_rf_CAPE["Actual"], label="Actual", color="blue")
plt.plot(
    results_rf_CAPE.index,
    results_rf_CAPE["Prediction"],
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




# Secondly the inv_CAPE
# Select Features and Target
features = ["inv_CAPE"]
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


    # Perform Hyperparameter Tuning using TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)

    best_score = float("inf")
    best_params = {}

    for n_estimators in param_grid["n_estimators"]:
        for max_depth in param_grid["max_depth"]:
            mse_cv = []  # Store validation scores for each fold

            for train_idx, val_idx in tscv.split(X_train):
                X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42,
                    n_jobs=-1  # Leverage parallelism
                )
                rf.fit(X_train_fold, y_train_fold)
                y_val_pred = rf.predict(X_val_fold)

                mse_fold = mean_squared_error(y_val_fold, y_val_pred)
                mse_cv.append(mse_fold)

            avg_mse = np.mean(mse_cv)

            if avg_mse < best_score:
                best_score = avg_mse
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

    # Train Final Model with Best Hyperparameters on Full Training Data
    best_rf = RandomForestRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
    best_rf.fit(X_train, y_train)

    # Test Final Model on Unseen Test Data
    y_pred = best_rf.predict(X_test)

    # Compute RMSE
    rmse_step = np.sqrt(mean_squared_error(y_test, y_pred))

    # Store results
    dates.append(date)
    actuals.append(y_test.values[0])
    predictions.append(y_pred[0])
    rmse_list.append(rmse_step)

    print(f"Date: {date.strftime('%Y-%m-%d')}, Best Params: {best_params}, RMSE: {rmse_step:.4f}")

# Save the results
results_df = pd.DataFrame({
    "Date": dates,
    "Prediction": predictions,
    "Actual": actuals,
    "RMSE": rmse_list,
}).set_index("Date")

results_df.to_csv(r"C:\Users\Käyttäjä\Desktop\Bachelor_proj\reports\final_results\final_results_rf_inv_CAPE.csv", index=True)


print(results_df["RMSE"].mean())

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





# Lastly the inv_CAPE with the improved CV
features = ["inv_CAPE"]
target = "10Y_stock_returns"
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, 20]
}
TRAIN_START = "1927-05-01"
TEST_START = "1960-01-01"
TEST_END = "2015-02-01"

# --- Preprocessing ---
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# --- Storage ---
dates = []
predictions = []
actuals = []
rmse_list = []
best_params_list = []

# --- Rolling Forecast ---
date_range = pd.date_range(TEST_START, TEST_END, freq="MS")

for date in date_range:
    train_end = (date - pd.DateOffset(years=10, months=1)).strftime("%Y-%m-%d")
    
    train_data = df.loc[TRAIN_START:train_end]
    test_data = df.loc[date:date]

    X_train = train_data[features].values
    y_train = train_data[target].values
    X_test = test_data[features].values
    y_test = test_data[target].values

    # Custom Grid Search with Your Expanding CV
    best_params, best_score, _ = custom_grid_search_cv(
        X_train, y_train,
        model_class=RandomForestRegressor,
        param_grid=param_grid,
        cv_func=rolling_origin_cv_rmse,
        horizon=120,
        min_train_size=120,
        step=12,
        val_window=12
    )

    # Train best model on full training set
    best_rf = RandomForestRegressor(**best_params, random_state=42)
    best_rf.fit(X_train, y_train)
    y_pred = best_rf.predict(X_test)[0]

    # Evaluate
    rmse_step = np.sqrt(mean_squared_error([y_test[0]], [y_pred]))

    # Store results
    best_params_list.append(best_params)
    dates.append(date)
    predictions.append(y_pred)
    actuals.append(y_test[0])
    rmse_list.append(rmse_step)

    print(f"Date: {date.date()}, Best Params: {best_params}, RMSE: {rmse_step:.4f}")

# Save to df
results_df = pd.DataFrame({
    "Date": dates,
    "Prediction": predictions,
    "Actual": actuals,
    "RMSE": rmse_list,
    "Best_Params": best_params_list
}).set_index("Date")

results_df.to_csv(r"C:\Users\Käyttäjä\Desktop\Bachelor_proj\reports\final_results\final_results_rf_inv_CAPE_improved.csv", index=True)


print(results_df["RMSE"].mean())

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
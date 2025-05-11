import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score

# Load the data
file_path = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/processed/merged_data.csv"
df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

# Let's first evaluate the SVR model with standard tscv
# Select features and target
df["inv_CAPE"] = 1 / df["CAPE"]
features = [
    "CPI",
    "inv_CAPE",
    "SVAR",
    "Real_10Y_Yield",
    "BondVol",
    "DFR",
]
target = "10Y_stock_returns"

# Standardize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Define training and test periods
train_start = "1927-05-01"
test_start = "1960-01-01"
test_end = "2015-02-01"

# Initialize storage lists
rmse_list = []
predictions = []
actuals = []
dates = []
# Parameter grid for SVR
param_grid = {
    "kernel": ["rbf"],  # Could add "linear", "poly", etc.
    "C": [0.1, 1, 10],  # Regularization strength
    "gamma": [0.01, 0.1, 1],  # Kernel coefficient for 'rbf'
    "epsilon": [0.01, 0.1, 0.5, 1.0],  # Tolerance margin
}
tscv = TimeSeriesSplit(n_splits=5)

# Rolling Forecast Loop
for date in pd.date_range(test_start, test_end, freq="MS"):  # Monthly steps
    train_end = (pd.Timestamp(date) - pd.DateOffset(years=10, months=1)).strftime(
        "%Y-%m-%d"
    )
    train = df.loc[train_start:train_end]  # Use past 10 years of data
    test_sample = df.loc[date:date]  # Predict one step ahead

    # Train SVR model
    X_train = train[features]
    y_train = train[target]

    # GridSearchCV (with 5-fold cross validation)
    svr_model = SVR()
    grid_search = GridSearchCV(
        estimator=svr_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    # Retrieve best model for this date
    best_svr = grid_search.best_estimator_

    # Prepare test sample
    X_test = test_sample[features]
    y_test = test_sample[target]
    # Predict next time step
    y_pred = best_svr.predict(X_test)[0]  # single test row => single value

    # Compute RMSE for this step
    rmse_step = np.sqrt(mean_squared_error([y_test], [y_pred]))

    # Store results
    dates.append(date)
    actuals.append(y_test.values[0])  # from single row
    predictions.append(y_pred)
    rmse_list.append(rmse_step)

# Convert results to DataFrame
results_df = pd.DataFrame(
    {"Date": dates, "Actual": actuals, "Predicted": predictions, "RMSE": rmse_list}
)

results_df.set_index("Date", inplace=True)

# To CSV
results_df.to_csv(
    "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/reports/results_SVR_multi.csv",
    index=True,
)

# Plot Predictions vs Actuals
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Actual"], label="Actual", color="blue")
plt.plot(
    results_df.index,
    results_df["Predicted"],
    label="Predicted (SVR)",
    color="red",
    linestyle="--",
)
plt.title("SVR Rolling Forecast (Hyperparameter Tuned)")
plt.xlabel("Date")
plt.ylabel("10-Year Annualized Return")
plt.legend()
plt.grid(True)
plt.show()


# Let's use only inverse CAPE
# Select features and target
df["inv_CAPE"] = 1 / df["CAPE"]
features = [
    "inv_CAPE",
]
target = "10Y_stock_returns"

# Standardize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Define training and test periods
train_start = "1927-05-01"
test_start = "1960-01-01"
test_end = "2015-02-01"

# Initialize storage
rmse_list = []
predictions = []
actuals = []
dates = []
best_params_list = []

# Parameter grid for SVR
param_grid = {
    "kernel": ["rbf"],  # Could add "linear", "poly", etc.
    "C": [0.1, 1, 10],  # Regularization strength
    "gamma": [0.01, 0.1, 1],
    "epsilon": [0.01, 0.1, 0.5, 1.0],  # Kernel coefficient for 'rbf'
}

tscv = TimeSeriesSplit(n_splits=5)
# Rolling Forecast Loop
for date in pd.date_range(test_start, test_end, freq="MS"):  # Monthly steps
    train_end = (pd.Timestamp(date) - pd.DateOffset(years=10, months=1)).strftime(
        "%Y-%m-%d"
    )
    train = df.loc[train_start:train_end]  # Use past 10 years of data
    test_sample = df.loc[date:date]  # Predict one step ahead

    # Train SVR model
    X_train = train[features]
    y_train = train[target]

    # GridSearchCV (with 5-fold cross validation)
    svr_model = SVR()
    grid_search = GridSearchCV(
        estimator=svr_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=tscv,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    # Retrieve best model for this date
    best_svr = grid_search.best_estimator_

    # Prepare test sample
    X_test = test_sample[features]
    y_test = test_sample[target]
    # Predict next time step
    y_pred = best_svr.predict(X_test)[0]  # single test row => single value

    # Compute RMSE for this step
    rmse_step = np.sqrt(mean_squared_error([y_test], [y_pred]))

    # Store results
    dates.append(date)
    actuals.append(y_test.values[0])  # from single row
    predictions.append(y_pred)
    rmse_list.append(rmse_step)
    print(f"Date: {date}, RMSE: {rmse_step}")

# Convert results to DataFrame
results_df = pd.DataFrame(
    {"Date": dates, "Actual": actuals, "Predicted": predictions, "RMSE": rmse_list}
)
results_df.set_index("Date", inplace=True)

# To CSV
results_df.to_csv(
    "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/reports/results_SVR_inv_CAPE.csv",
    index=True,
)


# Plot Predictions vs Actuals
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Actual"], label="Actual", color="blue")
plt.plot(
    results_df.index,
    results_df["Predicted"],
    label="Predicted (SVR)",
    color="red",
    linestyle="--",
)
plt.title("SVR Rolling Forecast (Hyperparameter Tuned)")
plt.xlabel("Date")
plt.ylabel("10-Year Annualized Return")
plt.legend()
plt.grid(True)
plt.show()


# Let's apply custom CV to the SVR model
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


# Let's use only CAPE
# Select features and target
df["inv_CAPE"] = 1 / df["CAPE"]
features = [
    "inv_CAPE",
]
target = "10Y_stock_returns"

# Standardize features
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# Define training and test periods
train_start = "1927-05-01"
test_start = "1960-01-01"
test_end = "2015-02-01"

# Initialize storage lists
rmse_list = []
predictions = []
actuals = []
dates = []
best_params_list = []

# Parameter grid for SVR
param_grid = {
    "kernel": ["rbf"],
    "C": [0.1, 1, 10],
    "gamma": [0.01, 0.1, 1],
    "epsilon": [0.01, 0.1, 0.5, 1.0],
}


# Rolling Forecast Loop
# Rolling Forecast Loop
for date in pd.date_range(test_start, test_end, freq="MS"):
    train_end = (pd.Timestamp(date) - pd.DateOffset(years=10, months=1)).strftime(
        "%Y-%m-%d"
    )
    train = df.loc[train_start:train_end]
    test_sample = df.loc[date:date]

    X_train = train[features].values
    y_train = train[target].values
    X_test = test_sample[features].values
    y_test = test_sample[target].values

    # Custom Grid Search using rolling_origin_cv_rmse
    best_params, best_score, _ = custom_grid_search_cv(
        X_train,
        y_train,
        model_class=SVR,
        param_grid=param_grid,
        cv_func=rolling_origin_cv_rmse,
        horizon=120,
        min_train_size=120,
        step=12,
        val_window=12,
    )

    # Fit the best SVR model to full training set
    best_svr = SVR(**best_params)
    best_svr.fit(X_train, y_train)

    # Predict next point
    y_pred = best_svr.predict(X_test)[0]
    rmse_step = np.sqrt(mean_squared_error([y_test[0]], [y_pred]))

    # Store results
    dates.append(date)
    actuals.append(y_test[0])
    predictions.append(y_pred)
    rmse_list.append(rmse_step)
    best_params_list.append(best_params)
    print(f"Date: {date}, RMSE: {rmse_step}, Best Params: {best_params}")

# Convert results to DataFrame
results_df = pd.DataFrame(
    {"Date": dates, "Actual": actuals, "Predicted": predictions, "RMSE": rmse_list}
)
results_df.set_index("Date", inplace=True)

# To CSV
results_df.to_csv(
    "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/reports/results_SVR_inv_CAPE_customcv.csv",
    index=True,
)

# Compute average RMSE
print(results_df["RMSE"].mean())
print(r2_score(results_df["Actual"], results_df["Predicted"]))

# Plot Predictions vs Actuals
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Actual"], label="Actual", color="blue")
plt.plot(
    results_df.index,
    results_df["Predicted"],
    label="Predicted (SVR)",
    color="red",
    linestyle="--",
)
plt.title("SVR Rolling Forecast (Hyperparameter Tuned)")
plt.xlabel("Date")
plt.ylabel("10-Year Annualized Return")
plt.legend()
plt.grid(True)
plt.show()

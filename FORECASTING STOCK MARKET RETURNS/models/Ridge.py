import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit


# Load the data
file_path = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/processed/merged_data.csv"
df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
# Convert the 10Y stock returns to a percentage
df["10Y_stock_returns"] = df["10Y_stock_returns"] * 100
# Define the inverse CAPE variable
df["inv_CAPE"] = 1 / df["CAPE"]


# Let's first evaluate the models using standard tscv
# Define features (X) and target (y)
features = [
    "CPI",
    "inv_CAPE",
    "SVAR",
    "Real_10Y_Yield",
    "BondVol",
    "DFR",
]
target = "10Y_stock_returns"


# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
# Initialize the Ridge regression model
df_scaled = df.copy()
df_scaled[features] = X_scaled  # Store scaled features

# Recursive Training Initialization
rmse_list = []
predictions = []
actuals = []
dates = []

alphas = [0.01, 0.1, 1, 10, 100]  # Range of alphas to test
tscv = TimeSeriesSplit(n_splits=5)

# Rolling Forecast Loop
for date in pd.date_range(
    "1960-01-01", "2015-02-01", freq="MS"
):  # Monthly rolling forecast
    train_end = (pd.Timestamp(date) - pd.DateOffset(years=10, months=1)).strftime(
        "%Y-%m-%d"
    )
    train = df_scaled.loc["1927-05-01":train_end]  # Use only past data
    test_sample = df_scaled.loc[date:date]  # Predict one step ahead

    # Split Features and Target
    X_train, y_train = train[features], train[target]
    X_test, y_test = test_sample[features], test_sample[target]

    # Ridge Hyperparameter Tuning
    ridge_cv = RidgeCV(alphas=alphas, cv=tscv)  # Cross-validation Ridge model
    ridge_cv.fit(X_train, y_train)  # Train Ridge Model

    # Predict Next Step
    y_pred = ridge_cv.predict(X_test)[0]  # Extract single prediction

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error([y_test], [y_pred]))
    rmse_list.append((date, rmse))

    # Store Predictions & Actuals
    predictions.append(y_pred)
    actuals.append(y_test.values[0])
    dates.append(date)

# Convert Results to DataFrame
results_df = pd.DataFrame(
    {
        "Date": dates,
        "Actual": actuals,
        "Predicted": predictions,
        "RMSE": [r[1] for r in rmse_list],
    }
)
results_df.set_index("Date", inplace=True)

# To CSV
results_df.to_csv(
    "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/reports/results_ridge_multi_tscv.csv",
    index=True,
)

# Plot Predictions vs Actuals
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Actual"], label="Actual")
plt.plot(results_df.index, results_df["Predicted"], label="Predicted")
plt.xlabel("Year")
plt.ylabel("10-Year Annualized Return")
plt.title("Ridge Regression Predictions vs Actuals")
plt.legend()
plt.grid(True)
plt.show()


# Let's use only the inverse of the CAPE ratio
features = ["inv_CAPE"]
target = "10Y_stock_returns"

# Standardize the features (Ridge is sensitive to feature scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Initialize the Ridge regression model
df_scaled = df.copy()
df_scaled[features] = X_scaled  # Store scaled features

# Recursive Training Initialization
rmse_list = []
predictions = []
actuals = []
dates = []

alphas = [0.01, 0.1, 1, 10, 100]  # Range of alphas to test
tscv = TimeSeriesSplit(n_splits=5)

# Rolling Forecast Loop
for date in pd.date_range(
    "1960-01-01", "2015-02-01", freq="MS"
):  # Monthly rolling forecast
    train_end = (pd.Timestamp(date) - pd.DateOffset(years=10, months=1)).strftime(
        "%Y-%m-%d"
    )
    train = df_scaled.loc["1927-05-01":train_end]  # Use only past data
    test_sample = df_scaled.loc[date:date]  # Predict one step ahead

    # Split Features and Target
    X_train, y_train = train[features], train[target]
    X_test, y_test = test_sample[features], test_sample[target]

    # Ridge Hyperparameter Tuning
    ridge_cv = RidgeCV(alphas=alphas, cv=tscv)
    # Cross-validation Ridge model
    ridge_cv.fit(X_train, y_train)  # Train Ridge Model

    # Predict Next Step
    y_pred = ridge_cv.predict(X_test)[0]  # Extract single prediction

    # Compute RMSE
    rmse = np.sqrt(mean_squared_error([y_test], [y_pred]))
    rmse_list.append((date, rmse))

    # Store Predictions & Actuals
    predictions.append(y_pred)
    actuals.append(y_test.values[0])
    dates.append(date)

# Convert Results to DataFrame
results_df = pd.DataFrame(
    {
        "Date": dates,
        "Actual": actuals,
        "Predicted": predictions,
        "RMSE": [r[1] for r in rmse_list],
    }
)
results_df.set_index("Date", inplace=True)

# To CSV
results_df.to_csv(
    "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/reports/results_ridge_inv_CAPE_tscv.csv",
    index=True,
)

# Plot Predictions vs Actuals
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Actual"], label="Actual")
plt.plot(results_df.index, results_df["Predicted"], label="Predicted")
plt.xlabel("Year")
plt.ylabel("10-Year Annualized Return")
plt.title("Ridge Regression Predictions vs Actuals")
plt.legend()
plt.grid(True)
plt.show()


# Let's initiate the model with the custom time series split
# A custom CV measure
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

        rmse_list, _, _ = cv_func(X, y, model, **cv_kwargs)
        mean_rmse = np.mean(rmse_list)

        results.append({**params, "mean_rmse": mean_rmse})

        if mean_rmse < best_score:
            best_score = mean_rmse
            best_params = params

    return best_params, best_score, results


# Let's initiate the model
# Preprocessing
df["inv_CAPE"] = 1 / df["CAPE"]
features = [
    "inv_CAPE",
]
target = "10Y_stock_returns"
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])
df_scaled = df.copy()
df_scaled[features] = X_scaled

# Hyperparameters
alphas = [0.01, 0.1, 1, 10, 100]

# Lists
rmse_list = []
predictions = []
actuals = []
dates = []
chosen_alphas = []

for date in pd.date_range("1960-01-01", "2015-02-01", freq="MS"):
    train_end = (pd.Timestamp(date) - pd.DateOffset(years=10, months=1)).strftime(
        "%Y-%m-%d"
    )
    train = df_scaled.loc["1927-05-01":train_end]
    test_sample = df_scaled.loc[date:date]

    X_train, y_train = train[features].values, train[target].values
    X_test, y_test = test_sample[features].values, test_sample[target].values

    best_alpha = None

    best_mae = float("inf")

    for alpha in alphas:
        model = Ridge(alpha=alpha)
        rmse_list_fold, mae_list_fold, _, _ = rolling_origin_cv_rmse(
            X_train,
            y_train,
            model,
            horizon=120,
            min_train_size=120,
            step=12,
            val_window=12,
        )

        mean_mae = np.mean(mae_list_fold)

        if mean_mae < best_mae:
            best_mae = mean_mae
            best_alpha = alpha

    # Retrain on full training set with best alpha
    model = Ridge(alpha=best_alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)[0]

    rmse = np.sqrt(mean_squared_error([y_test[0]], [y_pred]))

    predictions.append(y_pred)
    actuals.append(y_test[0])
    dates.append(date)
    rmse_list.append(rmse)
    chosen_alphas.append(best_alpha)

# Build results DataFrame
results_df = pd.DataFrame(
    {
        "Date": dates,
        "Actual": actuals,
        "Predicted": predictions,
        "RMSE": rmse_list,
        "BestAlpha": chosen_alphas,
    }
)
results_df.set_index("Date", inplace=True)

results_df.to_csv(
    "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/reports/results_ridge_inv_CAPE_customcv.csv",
    index=True,
)


# Plot Predictions vs Actuals
plt.figure(figsize=(12, 6))
plt.plot(results_df.index, results_df["Actual"], label="Actual")
plt.plot(results_df.index, results_df["Predicted"], label="Predicted")
plt.xlabel("Year")
plt.ylabel("10-Year Annualized Return")
plt.title("Ridge Regression Predictions vs Actuals")
plt.legend()
plt.grid(True)
plt.show()

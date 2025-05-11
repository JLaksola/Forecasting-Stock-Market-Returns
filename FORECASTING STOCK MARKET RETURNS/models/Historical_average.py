import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np


# Load your dataset
df = pd.read_csv(
    "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/processed/merged_data.csv",
    index_col="Date",
    parse_dates=True,
)

# Convert the target variable to percentage
df["10Y_stock_returns"] = df["10Y_stock_returns"] * 100

# Define forecast range
train_start = "1927-05-01"
test_start = "1960-01-01"
test_end = "2015-02-01"

# Initialize lists
naive_preds = []
actuals = []
dates = []
rmse_list = []

# Rolling forecast loop
for date in pd.date_range(test_start, test_end, freq="MS"):
    # Define training window
    train_end = pd.Timestamp(date) - pd.DateOffset(years=10, months=1)

    # Extract training data (realized forward 10Y returns)
    train = df.loc[train_start:train_end].dropna()

    # Forecast: mean of past forward returns
    naive_forecast = train["10Y_stock_returns"].mean()

    # True value (next forward return)
    if date in df.index and pd.notna(df.loc[date, "10Y_stock_returns"]):
        actual_return = df.loc[date, "10Y_stock_returns"]

        # Store results
        naive_preds.append(naive_forecast)
        actuals.append(actual_return)
        dates.append(date)

        # Compute RMSE for this prediction
        rmse = np.sqrt(mean_squared_error([actual_return], [naive_forecast]))
        rmse_list.append(rmse)

# Create results DataFrame
results_df = pd.DataFrame(
    {
        "Date": dates,
        "Naive_Predicted": naive_preds,
        "Actual": actuals,
        "RMSE": rmse_list,
    }
)
results_df.set_index("Date", inplace=True)


# To csv
results_df.to_csv(
    "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/reports/results_historical_average.csv",
    index=True,
)

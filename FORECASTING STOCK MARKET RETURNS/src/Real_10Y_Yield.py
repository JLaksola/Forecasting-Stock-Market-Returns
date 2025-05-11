import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg

file_path = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/interim/shillers_data_interim.csv"
df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
df.sort_index()

# Compute 12-month trailing (Year-over-Year) inflation rate
df["CPI"] = df["CPI"].pct_change(periods=12) * 100

# Convert to decimal
df["CPI_percent"] = df["CPI"] / 100

# Let's compute the syntetic expected inflation
# Define rolling window length (30 years of monthly data)
rolling_window = 30 * 12  # 30 years * 12 months

# Create a new column for expected 10-year inflation
df["Expected_Inflation"] = np.nan  # Placeholder

# Loop through the dataset, fitting an AR model and predicting next 10 years of inflation
for i in range(rolling_window + 12, len(df)):
    # Select past 30 years of data
    train_data = df["CPI_percent"].iloc[i - rolling_window : i]

    # Fit an Auto-Regressive model with 12 lags
    ar_model = AutoReg(train_data, lags=12, old_names=False).fit()

    # Forecast next 10 years (120 months) of inflation
    forecast_values = ar_model.predict(start=len(train_data), end=len(train_data) + 119)

    # Compute expected inflation as the average of the 10-year forecast
    df.loc[df.index[i], "Expected_Inflation"] = forecast_values.mean()


# Drop rows with NaN values (first 30 years of data won't have estimates)
df = df.dropna(subset=["Expected_Inflation"])

# Load 10-Year Nominal Treasury Yield (GS10) from your dataset
df["GS10"] = df["GS10"] / 100  # Convert % to decimal if necessary

# Compute Real 10-Year Bond Yield
df["Real_10Y_Yield"] = df["GS10"] - df["Expected_Inflation"]

# Keep only relevant columns
df = df[["CPI", "Real_10Y_Yield"]]

cleaned_data_path = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/interim/Real_10Y_Yield_and_CPI.csv"
df.to_csv(cleaned_data_path, index=True)

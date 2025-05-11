import pandas as pd

# Load the Shiller's data
file_path = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/raw/Shillers_data.xls"
df = pd.read_excel(file_path, header=[7])

# Drop the first row and reset the index
df = df.drop(columns=["Unnamed: 13", "Unnamed: 15"])

# Reset the index
df["Date"] = df["Date"].astype(str)

# Split year and month properly
df[["Year", "Month"]] = df["Date"].str.split(".", expand=True)

# Replace '1' with '10' ONLY if it's a single character (i.e., incorrectly coded)
df["Month"] = df["Month"].replace({"1": "10"})

# Recombine into a proper YYYY-MM format
df["Date"] = df["Year"] + "-" + df["Month"]

# Convert to datetime format
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m")

# Set as index
df.set_index("Date", inplace=True)

# Drop extra columns
df.drop(columns=["Year", "Month"], inplace=True)

# Rename columns
df[
    [
        "Price",
        "Dividend",
        "Earnings",
        "CPI",
        "GS10",
        "Real_Price",
        "Real_Dividend",
        "RTR_Price",
        "Real_Earnings",
        "Real_TR_Earnings",
        "CAPE",
        "TR_CAPE",
        "Excess_CAPE_Yield",
        "Month_total_bond_Returns",
        "Real_month_bond_Returns",
        "10Y_stock_returns",
        "10Y_bond_returns",
        "Real_10Y_excess_Returns",
    ]
] = df[
    [
        "Price",
        "Dividend",
        "Earnings",
        "CPI",
        "GS10",
        "Real_Price",
        "Real_Dividend",
        "RTR_Price",
        "Real_Earnings",
        "Real_TR_Earnings",
        "CAPE",
        "TR_CAPE",
        "Excess_CAPE_Yield",
        "Month_total_bond_Returns",
        "Real_month_bond_Returns",
        "10Y_stock_returns",
        "10Y_bond_returns",
        "Real_10Y_excess_Returns",
    ]
].apply(pd.to_numeric, errors="coerce")

# Let's save the cleaned data as interim data
cleaned_data_path = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/interim/shillers_data_interim.csv"
df.to_csv(cleaned_data_path, index=True)

# Let's compute treasury bill rates and volatility and variance of the returns from the Fama-French data
# Load the Fama-French data
# The data is in fixed width format, so we can use read_fwf
filepath1 = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/raw/F-F_Research_Data_Factors.txt"
filepath2 = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/raw/F-F_Research_Data_Factors_daily.txt"

df1 = pd.read_fwf(filepath1, skiprows=3)
df2 = pd.read_fwf(filepath2, skiprows=3)

# Let's define the date columns
# The first column is the date column, so we can use the first column as the date column
df1["Date"] = df1["Unnamed: 0"]
df2["Date"] = df2["Unnamed: 0"]

# Drop the first column
df1.drop(columns=["Unnamed: 0"], inplace=True)
df2.drop(columns=["Unnamed: 0"], inplace=True)

# Let's set the date column as the index
df1.set_index("Date", inplace=True)
df2.set_index("Date", inplace=True)

# Let's change the date dtype to integer
df1.index = df1.index.astype(int)
df2.index = df2.index.astype(int)

# Let's change the date to datetime format
df1.index = pd.to_datetime(df1.index, format="%Y%m")
df2.index = pd.to_datetime(df2.index, format="%Y%m%d")

# Let's then compute the daily volatility of the stock market
# First, we need to compute the daily returns
df2["Mkt"] = df2["Mkt-RF"] + df2["RF"]

# Then, we can compute the daily volatility
df2["ReturnVol"] = df2["Mkt"].rolling(window=252).std()

# Let's convert the daily volatility to monthly data
df_monthly = df2.resample("MS").first()[["ReturnVol"]]

# Let's compute the daily volatility of the bond market
df2["BondVol"] = df2["RF"].rolling(window=252).std()

# Let's convert the daily volatility to monthly data
df_monthly["BondVol"] = df2.resample("MS").first()["BondVol"]


# Let's compute the daily variance of the stock market
df2["RetunrVar"] = (df2["Mkt"]) ** 2
df2["SVAR"] = df2["RetunrVar"].rolling(window=252).sum()
df_monthly["SVAR"] = df2.resample("MS").first()["SVAR"]

# Let's import the RF as TBL
df_monthly["TBL"] = df2.resample("MS").first()["RF"]

# Let's make sure both indices are datetime and monthly-start
df.index = pd.to_datetime(df.index).to_period("M").to_timestamp()
df_monthly.index = pd.to_datetime(df_monthly.index).to_period("M").to_timestamp()

# Now join the two datasets
df = df.join(df_monthly, how="inner")

# Let's then compute the default return spread and the default yield spread
# Let's load the AAA and BAA data from FRED
aaa_path = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/raw/AAA.csv"
baa_path = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/raw/BAA.csv"
aaa_df = pd.read_csv(aaa_path, index_col="observation_date", parse_dates=True)
baa_df = pd.read_csv(baa_path, index_col="observation_date", parse_dates=True)
# Let's set the index to be the date
aaa_df.index.name = "Date"
baa_df.index.name = "Date"

# Trim to start from 1926-07-01
start_date = "1926-07-01"
aaa_df = aaa_df[aaa_df.index >= start_date]
baa_df = baa_df[baa_df.index >= start_date]

# DFY: BAA - AAA
df["DFY"] = baa_df["BAA"] - aaa_df["AAA"]

# DFR: BAA - GS10
df["DFR"] = baa_df["BAA"] - df["GS10"]


# Lastly, we need to import the CPI and the Real_10Y_Yield
file_path = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/interim/Real_10Y_Yield_and_CPI.csv"
df_cpi = pd.read_csv(file_path, index_col="Date", parse_dates=True)
df_cpi.index.name = "Date"
df_cpi = df_cpi[df_cpi.index >= start_date]
print(df_cpi)
# Merge the two datasets
# Drop the CPI column from df first
df = df.drop(columns="CPI")

# Now join the two datasets
df = df.join(df_cpi, how="left")
print(df.head())


# Save the final dataset
final_data_path = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/processed/merged_data.csv"
df.to_csv(final_data_path, index=True)

import pandas as pd
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox


# Load the data
file_path = "/Users/kayttaja/Desktop/FORECASTING STOCK MARKET RETURNS/data/processed/merged_data.csv"
df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
df["10Y_stock_returns"] = df["10Y_stock_returns"] * 100
df["inv_CAPE"] = 1 / df["CAPE"]
df["CAPE_pct_change"] = df["CAPE"].pct_change()
df = df.loc["1927-05-01":"2015-02-01"]


# Autolag ADF test
# Run ADF test for each column with automatic lag selection with AIC
# List of column names to test
columns_to_test = [
    "CPI",
    "CAPE",
    "inv_CAPE",
    "Real_10Y_Yield",
    "ReturnVol",
    "BondVol",
    "SVAR",
    "DFY",
    "DFR",
    "TBL",
    "10Y_stock_returns",
]

# Prepare list to collect results
results = []

for col in columns_to_test:
    series = df[col].dropna()
    result = adfuller(series, autolag="AIC")

    # Append results to list as a dictionary
    results.append(
        {
            "Variable": col,
            "ADF Statistic": round(result[0], 4),
            "p-value": round(result[1], 4),
            "Used Lag (AIC)": result[2],
            "Stationarity (10%)": "Stationary"
            if result[1] < 0.10
            else "Non-stationary",
        }
    )

# Convert list of dicts to DataFrame
summary_df = pd.DataFrame(results)
print(summary_df)

# Run ADF test for each column with automatic lag selection with BIC
# Prepare list to collect results
results = []

for col in columns_to_test:
    series = df[col].dropna()
    result = adfuller(series, autolag="BIC")

    # Append results to list as a dictionary
    results.append(
        {
            "Variable": col,
            "ADF Statistic": round(result[0], 4),
            "p-value": round(result[1], 4),
            "Used Lag (BIC)": result[2],
            "Stationarity (10%)": "Stationary"
            if result[1] < 0.10
            else "Non-stationary",
        }
    )

# Convert list of dicts to DataFrame
summary_df = pd.DataFrame(results)
# Print the summary DataFrame
print(summary_df)


# Let's use in-sample OLS regression to statistically evaluate the model
# Define features and target
features = [
    "CPI",
    "inv_CAPE",
    "Real_10Y_Yield",
    "ReturnVol",
    "BondVol",
    "SVAR",
    "DFY",
    "DFR",
    "TBL",
]

target = "10Y_stock_returns"
X_train, y_train = df[features], df[target]
# Add constant term for intercept
X_train = sm.add_constant(X_train)
# Fit OLS model
model = sm.OLS(y_train, X_train).fit()
# Print model summary
print(model.summary())

# Compute VIF
vif_data = pd.DataFrame()
vif_data["Variable"] = X_train.columns
vif_data["VIF"] = [
    variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])
]

# Display VIF results
print(vif_data)


# Check for multicollinearity
# Define thresholds for significance
def add_significance_stars(value):
    """Append '*' for moderate (|r| > 0.5), '**' for high (|r| > 0.75) correlation and '***' for extremely high (|r| > 0.9) correlation."""
    if abs(value) > 0.90:
        return f"{value:.2f}***"
    elif abs(value) > 0.70:
        return f"{value:.2f}**"  # High correlation
    elif abs(value) > 0.50:
        return f"{value:.2f}*"  # Moderate correlation
    else:
        return f"{value:.2f}"  # Low correlation


# Apply function to correlation matrix
correlation_matrix = X_train.corr()
correlation_matrix_stars = correlation_matrix.applymap(add_significance_stars)
print(correlation_matrix_stars)


# Define features and target
features = [
    "CPI",
    "inv_CAPE",
    "SVAR",
    "Real_10Y_Yield",
    "BondVol",
    "DFR",
]
target = "10Y_stock_returns"
X_train, y_train = df[features], df[target]

# Add constant term for intercept
X_train = sm.add_constant(X_train)
# Fit OLS model
model = sm.OLS(y_train, X_train).fit()
# Print model summary
print(model.summary())
residuals = model.resid

# Breusch-Pagan test for heteroskedasticity
bp_test = het_breuschpagan(residuals, X_train)

# Test results
bp_lm_stat = bp_test[0]
bp_p_value = bp_test[1]
bp_f_stat = bp_test[2]
bp_f_p_value = bp_test[3]

print(f"Breusch-Pagan Test:")
print(f"LM Statistic: {bp_lm_stat:.4f}, p-value: {bp_p_value:.4f}")
print(f"F-Statistic: {bp_f_stat:.4f}, p-value: {bp_f_p_value:.4f}")
print()

# Ljung-Box test for autocorrelation
ljung_box_test = acorr_ljungbox(residuals, lags=[10], return_df=True)

print("Ljung-Box Test (up to lag 10):")
print(ljung_box_test)

# Run the Newey-West adjusted standard errors
# Define features and target
n = len(df)
lag = int(n ** (1 / 3))
target = "10Y_stock_returns"
X_train, y_train = df[features], df[target]
# Add constant term for intercept
X_train = sm.add_constant(X_train)
# Fit OLS model
model = sm.OLS(y_train, X_train).fit()
# Newey-West standard errors
newey_west_se = model.get_robustcov_results(cov_type="HAC", maxlags=lag)
# Print model summary
print(newey_west_se.summary())

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree

# Load the dataset
filepath = "/Users/kayttaja/Desktop/Bachelor_proj/data/processed/merged_data.csv"
df = pd.read_csv(filepath, index_col="Date", parse_dates=True)

# Remove missing values in the target variable
df_cleaned = df.dropna(subset=["10Y_stock_returns"])

# Select the relevant columns
X = df_cleaned[["CAPE"]]  # Independent variable
y = df_cleaned["10Y_stock_returns"] * 100  # Dependent variable

# Train a Decision Tree Regressor
regressor = DecisionTreeRegressor(max_depth=3, random_state=42)
regressor.fit(X, y)

# Plot the decision tree
plt.figure(figsize=(12, 6))
tree.plot_tree(
    regressor,
    feature_names=["CAPE"],
    filled=True,
    impurity=False,
    rounded=True,
    precision=1,
)
plt.title("Regression Tree Model")
plt.show()

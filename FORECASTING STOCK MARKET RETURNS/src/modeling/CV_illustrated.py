import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import numpy as np

# Create a custom datetime index
dates = pd.date_range(start="1927-05-01", end="1949-12-01", freq="MS")
n_dates = len(dates)

# Adjust TimeSeriesSplit to match this length
tscv = TimeSeriesSplit(n_splits=5)

# Plot with dates on x-axis
fig, ax = plt.subplots(figsize=(12, 4))

for i, (train_index, test_index) in enumerate(tscv.split(np.arange(n_dates))):
    ax.scatter(
        dates[train_index],
        [i + 0.5] * len(train_index),
        color="blue",
        marker="s",
        label="Train" if i == 0 else "",
    )
    ax.scatter(
        dates[test_index],
        [i + 0.5] * len(test_index),
        color="orange",
        marker="s",
        label="Validate" if i == 0 else "",
    )

# Style the plot
ax.set_yticks(np.arange(1, 6))
ax.set_yticklabels([f"Fold {i}" for i in range(1, 6)])
ax.set_xlabel("Date")
ax.set_title("Time Series Split cross-validation (1927-1949)")
ax.legend(loc="lower right")
plt.grid()
plt.tight_layout()


# Suomeksi
# Create a custom datetime index
dates = pd.date_range(start="1927-05-01", end="1949-12-01", freq="MS")
n_dates = len(dates)

# Adjust TimeSeriesSplit to match this length
tscv = TimeSeriesSplit(n_splits=5)

# Plot with dates on x-axis
fig, ax = plt.subplots(figsize=(12, 4))

for i, (train_index, test_index) in enumerate(tscv.split(np.arange(n_dates))):
    ax.scatter(
        dates[train_index],
        [i + 0.5] * len(train_index),
        color="blue",
        marker="s",
        label="Harjoittelu" if i == 0 else "",
    )
    ax.scatter(
        dates[test_index],
        [i + 0.5] * len(test_index),
        color="orange",
        marker="s",
        label="Validiointi" if i == 0 else "",
    )

# Style the plot
ax.set_yticks(np.arange(1, 6))
ax.set_yticklabels([f"Jakso {i}" for i in range(1, 6)])
ax.set_xlabel("Päivämäärä")
ax.set_title("Aikajärjestyksen jakaminen (1927-1949)")
ax.legend(loc="lower right")
plt.grid()
plt.tight_layout()

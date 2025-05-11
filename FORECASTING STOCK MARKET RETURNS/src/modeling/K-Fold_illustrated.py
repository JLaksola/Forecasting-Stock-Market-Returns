import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

# Use date range for time index
dates = pd.date_range(start="1927-05-01", end="1949-12-01", periods=200)

# Standard K-Fold again with 5 folds
kf = KFold(n_splits=5, shuffle=False)

# Plot with actual dates instead of integer indices
fig, ax = plt.subplots(figsize=(12, 4))

for i, (train_idx, val_idx) in enumerate(kf.split(dates)):
    ax.plot(
        dates[train_idx],
        [i] * len(train_idx),
        color="blue",
        lw=6,
        label="Train" if i == 0 else "",
    )
    ax.plot(
        dates[val_idx],
        [i] * len(val_idx),
        color="orange",
        lw=6,
        label="Validate" if i == 0 else "",
    )

ax.set_yticks(range(5))
ax.set_yticklabels([f"Fold {i + 1}" for i in range(5)])
ax.set_xlabel("Date")
ax.set_title("K-Fold Cross-Validation(1927-1949)")
ax.legend(loc="lower right")
plt.grid()
plt.tight_layout()

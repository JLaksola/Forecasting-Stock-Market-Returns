# Re-import necessary modules after code execution environment reset
import matplotlib.pyplot as plt
import pandas as pd

# Parameters from the user's function
total_months = 300
min_train_size = 120
horizon = 120
val_window = 12
step = 12

# Create a time index from Jan 1927
dates = pd.date_range(start="1927-01-01", periods=total_months, freq="MS")
end_date = pd.Timestamp("1949-12-01")

# Plotting
fig, ax = plt.subplots(figsize=(12, 4))
fold = 0

for t in range(min_train_size, total_months - val_window - horizon + 1, step):
    train_start = 0
    train_end = t
    val_start = t + horizon
    val_end = val_start + val_window

    # Stop if validation set goes beyond cutoff
    if dates[val_end - 1] > end_date:
        break

    # Plot training (expanding)
    ax.plot(
        dates[train_start:train_end],
        [fold] * (train_end - train_start),
        color="blue",
        lw=6,
        label="Train" if fold == 0 else "",
    )
    # Plot validation
    ax.plot(
        dates[val_start:val_end],
        [fold] * (val_end - val_start),
        color="orange",
        lw=6,
        label="Validate" if fold == 0 else "",
    )

    fold += 1

# Final touches
ax.set_yticks(range(fold))
ax.set_yticklabels([f"Fold {i + 1}" for i in range(fold)])
ax.set_xlabel("Date")
ax.set_title("Customized Time Series Split Cross-Validation (1927-1949)")
ax.legend(loc="lower right")
plt.grid()
plt.tight_layout()


# Suomeksi
# Parameters from the user's function
total_months = 300
min_train_size = 120
horizon = 120
val_window = 12
step = 12

# Create a time index from Jan 1927
dates = pd.date_range(start="1927-01-01", periods=total_months, freq="MS")
end_date = pd.Timestamp("1949-12-01")

# Plotting
fig, ax = plt.subplots(figsize=(12, 4))
fold = 0

for t in range(min_train_size, total_months - val_window - horizon + 1, step):
    train_start = 0
    train_end = t
    val_start = t + horizon
    val_end = val_start + val_window

    # Stop if validation set goes beyond cutoff
    if dates[val_end - 1] > end_date:
        break

    # Plot training (expanding)
    ax.plot(
        dates[train_start:train_end],
        [fold] * (train_end - train_start),
        color="blue",
        lw=6,
        label="Harjoitus" if fold == 0 else "",
    )
    # Plot validation
    ax.plot(
        dates[val_start:val_end],
        [fold] * (val_end - val_start),
        color="orange",
        lw=6,
        label="Validointi" if fold == 0 else "",
    )

    fold += 1

# Final touches
ax.set_yticks(range(fold))
ax.set_yticklabels([f"Jakso {i + 1}" for i in range(fold)])
ax.set_xlabel("Päivämäärä")
ax.set_title("Mukautettu Aikajanan Jako Ristiinvalidoimiseen (1927-1949)")
ax.legend(loc="lower right")
plt.grid()
plt.tight_layout()
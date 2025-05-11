import numpy as np
import matplotlib.pyplot as plt

# Generate a synthetic dataset
np.random.seed(42)
data = np.random.normal(loc=50, scale=15, size=100)  # 100 data points

# Number of bootstrap samples
num_samples = 2

# Create bootstrap samples
bootstrap_samples = [
    np.random.choice(data, size=len(data), replace=True) for _ in range(num_samples)
]
colors = ["red", "green"]

# Plot each bootstrap sample in separate graphs
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# Set the title of the figure
fig.suptitle("Bootstrap Samples From Original Data", fontsize=16, fontweight="bold")
# Flatten the axes array for easy iteration
axes = axes.flatten()

# Plot original data
axes[0].hist(data, bins=15, alpha=0.5, color="blue", edgecolor="black")
axes[0].set_title("Original Data")
axes[0].set_xlabel("Value")
axes[0].set_ylabel("Frequency")

# Plot each bootstrap sample separately
for i, sample in enumerate(bootstrap_samples):
    axes[i + 1].hist(sample, bins=15, alpha=0.7, color=colors[i], edgecolor="black")
    axes[i + 1].set_title(f"Bootstrap Sample {i + 1}")
    axes[i + 1].set_xlabel("Value")
    axes[i + 1].set_ylabel("Frequency")

# Remove any extra subplots (if more than necessary)
for j in range(len(bootstrap_samples) + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
plt.show()

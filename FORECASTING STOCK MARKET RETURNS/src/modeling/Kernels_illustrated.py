import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Generate synthetic non-linear data
np.random.seed(42)
X = np.sort(5 * np.random.rand(50, 1), axis=0)  # 50 data points
y = np.sin(X).ravel() + 0.1 * np.random.randn(50)  # Non-linear function with noise

# Fit different SVR models
svr_linear = SVR(kernel="linear", C=1e3, epsilon=0.1)
svr_poly = SVR(kernel="poly", degree=3, C=1e3, epsilon=0.1)
svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.5, epsilon=0.1)

svr_linear.fit(X, y)
svr_poly.fit(X, y)
svr_rbf.fit(X, y)

# Predict values for plotting
X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
y_linear = svr_linear.predict(X_plot)
y_poly = svr_poly.predict(X_plot)
y_rbf = svr_rbf.predict(X_plot)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Linear Kernel SVR
axes[0].scatter(X, y, color="black", label="Data points")
axes[0].plot(X_plot, y_linear, color="blue", lw=2, label="Linear SVR")
axes[0].set_title("Linear Kernel SVR")
axes[0].legend()
axes[0].grid()

# Polynomial Kernel SVR
axes[1].scatter(X, y, color="black", label="Data points")
axes[1].plot(
    X_plot, y_poly, color="red", lw=2, label="Polynomial Kernel SVR (degree=3)"
)
axes[1].set_title("Polynomial Kernel SVR")
axes[1].legend()
axes[1].grid()

# RBF Kernel SVR
axes[2].scatter(X, y, color="black", label="Data points")
axes[2].plot(X_plot, y_rbf, color="green", lw=2, label="RBF Kernel SVR")
axes[2].set_title("RBF Kernel SVR")
axes[2].legend()
axes[2].grid()

plt.show()


# Fit RBF SVR models with different C values
# Generate synthetic non-linear data
np.random.seed(42)
X = np.sort(5 * np.random.rand(50, 1), axis=0)  # 50 data points
y = np.sin(X).ravel() + 0.1 * np.random.randn(50)  # Non-linear function with noise

# Fit RBF SVR models with different C values
svr_low_C = SVR(kernel="rbf", C=0.1, gamma=0.5, epsilon=0.1)  # Low C (Flexible fit)
svr_medium_C = SVR(kernel="rbf", C=1, gamma=0.5, epsilon=0.1)  # Medium C (Balanced)
svr_high_C = SVR(kernel="rbf", C=1000, gamma=0.5, epsilon=0.1)  # High C (Strict fit)

svr_low_C.fit(X, y)
svr_medium_C.fit(X, y)
svr_high_C.fit(X, y)

# Predict values for plotting
X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
y_low_C = svr_low_C.predict(X_plot)
y_medium_C = svr_medium_C.predict(X_plot)
y_high_C = svr_high_C.predict(X_plot)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Low C (More Tolerance)
axes[0].scatter(X, y, color="black", label="Data points")
axes[0].plot(X_plot, y_low_C, color="blue", lw=2, label="SVR (Low C)")
axes[0].set_title("SVR with Low C (Flexible Fit)")
axes[0].legend()
axes[0].grid()

# Medium C (Balanced)
axes[1].scatter(X, y, color="black", label="Data points")
axes[1].plot(X_plot, y_medium_C, color="red", lw=2, label="SVR (Medium C)")
axes[1].set_title("SVR with Medium C (Balanced Fit)")
axes[1].legend()
axes[1].grid()

# High C (Strict Fit)
axes[2].scatter(X, y, color="black", label="Data points")
axes[2].plot(X_plot, y_high_C, color="green", lw=2, label="SVR (High C)")
axes[2].set_title("SVR with High C (Strict Fit)")
axes[2].legend()
axes[2].grid()

plt.show()


# Fit RBF SVR models with different gamma values
# Generate synthetic non-linear data
np.random.seed(42)
X = np.sort(5 * np.random.rand(50, 1), axis=0)  # 50 data points
y = np.sin(X).ravel() + 0.1 * np.random.randn(50)  # Non-linear function with noise

# Fit RBF SVR models with different gamma values
svr_low_gamma = SVR(
    kernel="rbf", C=10, gamma=0.01, epsilon=0.1
)  # Low gamma (Broad Influence)
svr_medium_gamma = SVR(
    kernel="rbf", C=10, gamma=0.5, epsilon=0.1
)  # Medium gamma (Balanced)
svr_high_gamma = SVR(
    kernel="rbf", C=10, gamma=5, epsilon=0.1
)  # High gamma (Narrow Influence)

svr_low_gamma.fit(X, y)
svr_medium_gamma.fit(X, y)
svr_high_gamma.fit(X, y)

# Predict values for plotting
X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
y_low_gamma = svr_low_gamma.predict(X_plot)
y_medium_gamma = svr_medium_gamma.predict(X_plot)
y_high_gamma = svr_high_gamma.predict(X_plot)

# Plot results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Low Gamma (Broad Influence)
axes[0].scatter(X, y, color="black", label="Data points")
axes[0].plot(X_plot, y_low_gamma, color="blue", lw=2, label=r"SVR ($\gamma = 0.01$)")
axes[0].set_title(r"SVR with Low $\gamma$ (Broad Influence)")
axes[0].legend()
axes[0].grid()

# Medium Gamma (Balanced)
axes[1].scatter(X, y, color="black", label="Data points")
axes[1].plot(X_plot, y_medium_gamma, color="red", lw=2, label=r"SVR ($\gamma = 0.5$)")
axes[1].set_title(r"SVR with Medium $\gamma$ (Balanced)")
axes[1].legend()
axes[1].grid()

# High Gamma (Narrow Influence)
axes[2].scatter(X, y, color="black", label="Data points")
axes[2].plot(X_plot, y_high_gamma, color="green", lw=2, label=r"SVR ($\gamma = 5$)")
axes[2].set_title(r"SVR with High $\gamma$ (Narrow Influence)")
axes[2].legend()
axes[2].grid()

plt.show()


# Generate sample data (nonlinear function)
np.random.seed(42)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.2, X.shape[0])  # Adding noise

# Define different gamma values
gamma_values = [0.1, 1, 10]

# Create subplots
plt.figure(figsize=(12, 4))

for i, gamma in enumerate(gamma_values):
    # Fit SVR model with RBF kernel
    svr = SVR(kernel="rbf", C=100, gamma=gamma, epsilon=0.1)
    svr.fit(X, y)

    # Predict
    X_test = np.linspace(0, 5, 100).reshape(-1, 1)
    y_pred = svr.predict(X_test)

    # Plot
    plt.subplot(1, 3, i + 1)
    plt.scatter(X, y, color="red", label="Data")
    plt.plot(X_test, y_pred, color="blue", lw=2, label=f"SVR (gamma={gamma})")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.title(f"SVR with gamma={gamma}")
    plt.grid()

# Show plot
plt.tight_layout()
plt.show()

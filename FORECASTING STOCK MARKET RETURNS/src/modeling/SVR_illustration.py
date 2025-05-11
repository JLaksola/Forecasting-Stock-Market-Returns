import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR

# Generate synthetic continuous and positive data
np.random.seed(42)
X = np.sort(5 * np.random.rand(15, 1), axis=0)  # 50 data points, all positive
y = np.log(X + 1).ravel()  # A continuous and positive function (log transformation)

# Hard Margin SVR: Larger epsilon for wider margin, large C for strict fit
svr_hard = SVR(kernel="linear", C=1e6, epsilon=0.5)  # Larger epsilon for wider margin
svr_hard.fit(X, y)

# Predict values
X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
y_pred = svr_hard.predict(X_plot)

# Get support vectors
support_vectors = X[svr_hard.support_]

# Plot the results
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color="green", edgecolors="black", s=50, label="Data points")
plt.plot(X_plot, y_pred, color="blue", lw=2, label="SVR Prediction")

# Plot wider epsilon margin
plt.plot(X_plot, y_pred + svr_hard.epsilon, "k--", lw=1, label=r"$\epsilon$-Margin")
plt.plot(X_plot, y_pred - svr_hard.epsilon, "k--", lw=1)

# Labels and title
plt.xlabel("X")
plt.ylabel("y")
plt.title("Hard Margin SVR")
plt.legend()
plt.grid()
plt.show()


# Generate synthetic dataset (continuous and positive)
np.random.seed(42)
X = np.sort(5 * np.random.rand(50, 1), axis=0)  # 50 continuous positive data points
y = np.log(X + 1).ravel() + 0.1 * np.random.randn(50)  # Log function with slight noise

# Define Hard Margin SVR (Strict Fit)
svr_hard = SVR(
    kernel="linear", C=1e6, epsilon=0.01
)  # Very small epsilon for strict margin
svr_hard.fit(X, y)

# Define Soft Margin SVR (Allows Some Error)
svr_soft = SVR(
    kernel="linear", C=1, epsilon=0.3
)  # Higher epsilon allows margin violations
svr_soft.fit(X, y)

# Generate predictions
X_plot = np.linspace(0, 5, 100).reshape(-1, 1)
y_pred_hard = svr_hard.predict(X_plot)
y_pred_soft = svr_soft.predict(X_plot)

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Hard Margin SVR Plot
axes[0].scatter(X, y, color="green", edgecolors="black", s=50, label="Data points")
axes[0].plot(
    X_plot, y_pred_hard, color="blue", lw=2, label="Hard Margin SVR Prediction"
)
axes[0].plot(
    X_plot, y_pred_hard + svr_hard.epsilon, "k--", lw=1, label=r"$\epsilon$-Margin"
)
axes[0].plot(X_plot, y_pred_hard - svr_hard.epsilon, "k--", lw=1)
axes[0].set_title("Hard Margin SVR (Strict Fit)")
axes[0].legend()
axes[0].grid()

# Soft Margin SVR Plot
axes[1].scatter(X, y, color="green", edgecolors="black", s=50, label="Data points")
axes[1].plot(X_plot, y_pred_soft, color="red", lw=2, label="Soft Margin SVR Prediction")
axes[1].plot(
    X_plot, y_pred_soft + svr_soft.epsilon, "k--", lw=1, label=r"$\epsilon$-Margin"
)
axes[1].plot(X_plot, y_pred_soft - svr_soft.epsilon, "k--", lw=1)
axes[1].set_title("Soft Margin SVR (Allows Some Errors)")
axes[1].legend()
axes[1].grid()

plt.show()


# Generate random data points
np.random.seed(42)
X = np.linspace(0, 10, 15)
y = 2 * X + 1 + np.random.randn(15) * 3.5  # Linear relation with noise

# Define SVR model parameters
w = 2
b = 1
epsilon = 3  # Epsilon margin

# Compute the regression line and margins
y_pred = w * X + b
upper_margin = y_pred + epsilon
lower_margin = y_pred - epsilon

# Compute slack variables (deviations beyond epsilon margin)
slack = np.zeros_like(y)
above_margin = y > upper_margin
below_margin = y < lower_margin

slack[above_margin] = y[above_margin] - upper_margin[above_margin]
slack[below_margin] = lower_margin[below_margin] - y[below_margin]

# Plot the SVR model
plt.figure(figsize=(8, 6))
plt.plot(X, y_pred, "b-", label=r"$y = w^Tx + b$")  # Regression line
plt.plot(X, upper_margin, "b--", label=r"$+\epsilon$ margin")  # Upper margin
plt.plot(X, lower_margin, "b--", label=r"$-\epsilon$ margin")  # Lower margin

# Plot data points
plt.scatter(X, y, color="green", edgecolors="black", marker=".", label="Data points")

# Plot slack variables (ξ) for points outside the margin
for i in range(len(X)):
    if above_margin[i]:  # Above the upper margin
        plt.plot([X[i], X[i]], [upper_margin[i], y[i]], "r-", lw=1.5)
        plt.text(
            X[i],
            (y_pred[i] + epsilon + y[i]) / 2,
            r"$\zeta$",
            fontsize=12,
            color="blue",
        )
    elif below_margin[i]:  # Below the lower margin
        plt.plot([X[i], X[i]], [y[i], lower_margin[i]], "r-", lw=1.5)
        plt.text(
            X[i],
            (y_pred[i] - epsilon + y[i]) / 2,
            r"$\zeta$*",
            fontsize=12,
            color="blue",
        )

# Labels and legend
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.legend()
plt.title("Soft Margin Support Vector Regression (SVR)")

# Show plot
plt.show()

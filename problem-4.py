import numpy as np
import pickle
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def regressionObjVal(w, X, y, lambd):
    """
    Compute the regularized squared loss and its gradient for Ridge Regression:
    J(w) = 1/2 * (y - Xw)^T * (y - Xw) + 1/2 * λ * w^T * w

    Parameters:
    w - Weight vector (flattened)
    X - Feature matrix (N x d)
    y - Target vector (N x 1)
    lambd - Regularization parameter (scalar)

    Returns:
    error - Regularized squared loss (scalar)
    error_grad - Gradient of the loss with respect to w (d x 1)
    """
    w = w.reshape(-1, 1)  # Ensure w is a column vector
    residuals = y - X @ w  # Residuals (y - Xw)

    # Regularized loss
    error = 0.5 * (residuals.T @ residuals + lambd * w.T @ w).item()

    # Gradient of the loss
    error_grad = -(X.T @ residuals) + lambd * w
    return error, error_grad.flatten()  # Flatten gradient for compatibility with scipy.optimize.minimize

# Load the diabetes dataset
with open('diabetes.pickle', 'rb') as f:
    data = pickle.load(f, encoding='latin1')
X, y, Xtest, ytest = data

# Add intercept to feature matrices
X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
Xtest_with_intercept = np.hstack((np.ones((Xtest.shape[0], 1)), Xtest))

# Set λ values for regularization
lambdas = np.linspace(0, 1, 101)  # λ values from 0 to 1 in steps of 0.01

# Initialize lists to store training and test errors
train_errors = []
test_errors = []

# Optimize weights for each λ using scipy.optimize.minimize
for lambd in lambdas:
    # Initial weight vector (all zeros)
    w_init = np.zeros((X_with_intercept.shape[1], 1))

    # Minimize the objective function
    args = (X_with_intercept, y, lambd)
    result = minimize(
        fun=lambda w: regressionObjVal(w, *args),  # Objective function
        x0=w_init.flatten(),  # Initial weights
        jac=True,  # Provide gradient (Jacobian)
        method='CG',  # Conjugate Gradient method
    )

    # Extract optimized weights
    w_optimized = result.x.reshape(-1, 1)

    # Compute training and test errors
    train_mse = np.mean((y - X_with_intercept @ w_optimized) ** 2)
    test_mse = np.mean((ytest - Xtest_with_intercept @ w_optimized) ** 2)

    # Append errors to the lists
    train_errors.append(train_mse)
    test_errors.append(test_mse)

# Plot the errors
plt.figure(figsize=(10, 6))
plt.plot(lambdas, train_errors, label='Training Error')
plt.plot(lambdas, test_errors, label='Testing Error')
plt.xlabel('λ (Regularization Parameter)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training and Testing Errors vs λ (Gradient Descent)')
plt.legend()
plt.grid(True)
plt.show()

# Find the optimal λ (the one with the lowest test error)
optimal_lambda = lambdas[np.argmin(test_errors)]
print(f"Optimal λ: {optimal_lambda:.2f}")
print(f"Minimum Test Error: {min(test_errors):.4f}")

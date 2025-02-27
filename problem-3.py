import numpy as np
import pickle
import matplotlib.pyplot as plt

def learnRidgeRegression(X, y, lambd):
    """
    Compute Ridge Regression weights using the formula:
    w = (X^T * X + λ * I)^(-1) * X^T * y

    Parameters:
    X - Feature matrix (N x d)
    y - Target vector (N x 1)
    lambd - Regularization parameter (scalar)

    Returns:
    w - Weight vector (d x 1)
    """
    d = X.shape[1]  # Number of features
    I = np.eye(d)  # Identity matrix of size d x d
    w = np.linalg.inv(X.T @ X + lambd * I) @ X.T @ y
    return w

def testOLERegression(w, Xtest, ytest):
    """
    Compute Mean Squared Error (MSE) for predictions.

    Parameters:
    w - Weight vector (d x 1)
    Xtest - Test feature matrix (N x d)
    ytest - Test target vector (N x 1)

    Returns:
    mse - Mean Squared Error (scalar)
    """
    ypred = Xtest @ w  # Compute predictions
    mse = np.mean((ytest - ypred) ** 2)  # Compute MSE
    return mse

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
weights_ridge = []  # Store weights for Ridge Regression
weights_ole = learnRidgeRegression(X_with_intercept, y, 0)  # OLE weights with λ=0

# Print OLE weights
print("\nWeights (OLE - λ = 0):")
print(weights_ole.flatten())

# Loop through λ values
for lambd in lambdas:
    # Compute Ridge Regression weights
    w_ridge = learnRidgeRegression(X_with_intercept, y, lambd)
    weights_ridge.append(w_ridge.flatten())  # Store weights

    # Compute training and test errors
    train_mse = testOLERegression(w_ridge, X_with_intercept, y)
    test_mse = testOLERegression(w_ridge, Xtest_with_intercept, ytest)

    # Append errors to the lists
    train_errors.append(train_mse)
    test_errors.append(test_mse)

# Plot the errors
plt.figure(figsize=(10, 6))
plt.plot(lambdas, train_errors, label='Training Error')
plt.plot(lambdas, test_errors, label='Testing Error')
plt.xlabel('λ (Regularization Parameter)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Training and Testing Errors vs λ')
plt.legend()
plt.grid(True)
plt.show()

# Find the optimal λ (the one with the lowest test error)
optimal_lambda = lambdas[np.argmin(test_errors)]
print(f"\nOptimal λ: {optimal_lambda:.2f}")
print(f"Minimum Test Error: {min(test_errors):.4f}")

# Print weights for Ridge Regression for the optimal λ
optimal_weights = weights_ridge[np.argmin(test_errors)]
print("\nWeights (Ridge Regression - Optimal λ):")
print(optimal_weights)

# Optional: Compare magnitudes of weights (OLE vs Ridge Regression)
print("\nWeight Comparison (Absolute Values):")
print("Feature Index | OLE Weight | Ridge Weight (Optimal λ)")
for i, (ole_w, ridge_w) in enumerate(zip(weights_ole.flatten(), optimal_weights)):
    print(f"{i:13} | {ole_w:10.6f} | {ridge_w:18.6f}")

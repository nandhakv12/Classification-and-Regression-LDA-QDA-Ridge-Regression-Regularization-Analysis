import numpy as np
import pickle
import matplotlib.pyplot as plt


def mapNonLinear(x, p):
    """
    Maps input x into a polynomial feature vector of degree p:
    [1, x, x^2, ..., x^p]

    Parameters:
    x - Input column vector (N x 1)
    p - Degree of polynomial (integer)

    Returns:
    Xp - Polynomial feature matrix (N x (p+1))
    """
    N = x.shape[0]  # Number of samples
    Xp = np.ones((N, p + 1))  # Initialize with ones for the bias term (x^0)

    for degree in range(1, p + 1):
        Xp[:, degree] = x.flatten() ** degree  # Compute x^degree for each column

    return Xp


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

# Use the third feature (index 2) as the input
x_train = X[:, 2].reshape(-1, 1)
x_test = Xtest[:, 2].reshape(-1, 1)

# Lambda values
lambda_0 = 0  # No regularization
optimal_lambda = 0.06

# Polynomial degrees to test
p_values = range(7)  # p from 0 to 6

# Store train and test errors for different polynomial degrees
train_errors_lambda_0 = []
test_errors_lambda_0 = []
train_errors_optimal_lambda = []
test_errors_optimal_lambda = []

for p in p_values:
    # Map to polynomial features of degree p
    Xp_train = mapNonLinear(x_train, p)
    Xp_test = mapNonLinear(x_test, p)

    # Train and evaluate for lambda = 0
    w_lambda_0 = learnRidgeRegression(Xp_train, y, lambda_0)
    train_mse_lambda_0 = testOLERegression(w_lambda_0, Xp_train, y)
    test_mse_lambda_0 = testOLERegression(w_lambda_0, Xp_test, ytest)

    # Train and evaluate for optimal lambda
    w_optimal_lambda = learnRidgeRegression(Xp_train, y, optimal_lambda)
    train_mse_optimal_lambda = testOLERegression(w_optimal_lambda, Xp_train, y)
    test_mse_optimal_lambda = testOLERegression(w_optimal_lambda, Xp_test, ytest)

    # Store errors
    train_errors_lambda_0.append(train_mse_lambda_0)
    test_errors_lambda_0.append(test_mse_lambda_0)
    train_errors_optimal_lambda.append(train_mse_optimal_lambda)
    test_errors_optimal_lambda.append(test_mse_optimal_lambda)

# Plot the errors for both lambda = 0 and optimal lambda
plt.figure(figsize=(12, 6))

# Plot for lambda = 0
plt.subplot(1, 2, 1)
plt.plot(p_values, train_errors_lambda_0, label='Training Error (λ=0)')
plt.plot(p_values, test_errors_lambda_0, label='Testing Error (λ=0)')
plt.xlabel('Degree of Polynomial (p)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Errors for λ=0 (No Regularization)')
plt.legend()
plt.grid(True)

# Plot for optimal lambda
plt.subplot(1, 2, 2)
plt.plot(p_values, train_errors_optimal_lambda, label=f'Training Error (λ={optimal_lambda})')
plt.plot(p_values, test_errors_optimal_lambda, label=f'Testing Error (λ={optimal_lambda})')
plt.xlabel('Degree of Polynomial (p)')
plt.ylabel('Mean Squared Error (MSE)')
plt.title(f'Errors for Optimal λ={optimal_lambda}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Find the optimal p for each lambda setting
optimal_p_lambda_0 = p_values[np.argmin(test_errors_lambda_0)]
optimal_p_optimal_lambda = p_values[np.argmin(test_errors_optimal_lambda)]
print(f"Optimal p for λ=0: {optimal_p_lambda_0}")
print(f"Optimal p for λ={optimal_lambda}: {optimal_p_optimal_lambda}")
# Print all errors for both lambda settings
print("Degree of Polynomial (p) | Train Error (λ=0) | Test Error (λ=0) | Train Error (λ=0.06) | Test Error (λ=0.06)")
print("-" * 80)

for p, train_err_0, test_err_0, train_err_opt, test_err_opt in zip(
        p_values, train_errors_lambda_0, test_errors_lambda_0, train_errors_optimal_lambda, test_errors_optimal_lambda):
    print(f"{p:<24} | {train_err_0:<16.4f} | {test_err_0:<15.4f} | {train_err_opt:<19.4f} | {test_err_opt:<15.4f}")

# Also print optimal p values
print("\nSummary:")
print(f"Optimal p for λ=0: {optimal_p_lambda_0}")
print(f"Optimal p for λ={optimal_lambda}: {optimal_p_optimal_lambda}")

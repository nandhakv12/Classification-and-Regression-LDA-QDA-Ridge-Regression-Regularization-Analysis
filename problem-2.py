import numpy as np
import pickle
import sys

def learnOLERegression(X, y):
    # Inputs:
    # X = N x d
    # y = N x 1
    # Output:
    # w = d x 1

    # Compute the ordinary least squares estimator
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xt_y = X.T @ y
    w = XtX_inv @ Xt_y
    return w

def testOLERegression(w, Xtest, ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = N x 1
    # Output:
    # mse

    # Compute predictions
    y_pred = Xtest @ w
    # Compute mean squared error
    mse = np.mean((ytest - y_pred) ** 2)
    return mse


# Main script
# Load the diabetes data
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('diabetes.pickle', 'rb'), encoding='latin1')

# Case 1: Without intercept
print("Case 1: Without Intercept")
w_no_intercept = learnOLERegression(X, y)
mse_train_no_intercept = testOLERegression(w_no_intercept, X, y)
mse_test_no_intercept = testOLERegression(w_no_intercept, Xtest, ytest)
print(f"MSE (Train, No Intercept): {mse_train_no_intercept:.4f}")
print(f"MSE (Test, No Intercept): {mse_test_no_intercept:.4f}")

# Case 2: With intercept
print("\nCase 2: With Intercept")
# Add a column of ones to the feature matrices for the intercept term
X_with_intercept = np.hstack((np.ones((X.shape[0], 1)), X))
Xtest_with_intercept = np.hstack((np.ones((Xtest.shape[0], 1)), Xtest))

w_with_intercept = learnOLERegression(X_with_intercept, y)
mse_train_with_intercept = testOLERegression(w_with_intercept, X_with_intercept, y)
mse_test_with_intercept = testOLERegression(w_with_intercept, Xtest_with_intercept, ytest)  # Corrected

print(f"MSE (Train, With Intercept): {mse_train_with_intercept:.4f}")
print(f"MSE (Test, With Intercept): {mse_test_with_intercept:.4f}")

# Compare results
if mse_test_with_intercept < mse_test_no_intercept:
    print("\nUsing the intercept gives better results on the test set.")
else:
    print("\nNot using the intercept gives better results on the test set.")

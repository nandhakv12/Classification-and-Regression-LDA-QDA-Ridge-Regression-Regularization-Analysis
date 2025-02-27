import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys


def ldaLearn(X, y):
    # Calculate class means and shared covariance matrix
    classes = np.unique(y)
    d = X.shape[1]  # Number of features
    k = len(classes)  # Number of classes

    # Calculate means for each class
    means = np.zeros((d, k))
    for i, cls in enumerate(classes):
        means[:, i] = X[y.flatten() == cls].mean(axis=0)

    # Calculate shared covariance matrix
    covmat = np.cov(X, rowvar=False)

    return means, covmat

def qdaLearn(X, y):
    # Calculate class-specific means and covariance matrices
    classes = np.unique(y)
    d = X.shape[1]  # Number of features
    k = len(classes)  # Number of classes

    # Calculate means and covariance matrices for each class
    means = np.zeros((d, k))
    covmats = []
    for i, cls in enumerate(classes):
        X_cls = X[y.flatten() == cls]
        means[:, i] = X_cls.mean(axis=0)
        covmats.append(np.cov(X_cls, rowvar=False))

    return means, covmats

def ldaTest(means, covmat, Xtest, ytest):
    # Predict using LDA
    covmat_inv = np.linalg.inv(covmat)

    # Calculate discriminants for each class
    means_inv_cov = means.T @ covmat_inv
    const_terms = -0.5 * np.sum(means_inv_cov * means.T, axis=1)

    # Calculate discriminant scores
    discriminants = Xtest @ means_inv_cov.T + const_terms

    # Predict class with the highest score
    ypred = np.argmax(discriminants, axis=1) + 1  # Classes start from 1
    acc = np.mean(ypred == ytest.flatten()) * 100

    return acc, ypred.reshape(-1, 1)

def qdaTest(means, covmats, Xtest, ytest):
    # Predict using QDA
    discriminants = []

    for i, mean in enumerate(means.T):
        covmat = covmats[i]
        covmat_inv = np.linalg.inv(covmat)
        covmat_det = np.linalg.det(covmat)

        diff = Xtest - mean
        term1 = -0.5 * np.sum(diff @ covmat_inv * diff, axis=1)
        term2 = -0.5 * np.log(covmat_det)
        discriminants.append(term1 + term2)
    discriminants = np.array(discriminants).T

    # Predict class with the largest discriminant
    ypred = np.argmax(discriminants, axis=1) + 1  # Adding 1 for class labels
    acc = np.mean(ypred == ytest.flatten()) * 100

    return acc, ypred.reshape(-1, 1)

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


def regressionObjVal(w, X, y, lambd):
    """
    Compute the regularized squared loss and its gradient for Ridge Regression:
    J(w) = 1/2 * (y - Xw)^T * (y - Xw) + 1/2 * λ * w^T * w

    Parameters:
    w - Weight vector (flattened or column vector)
    X - Feature matrix (N x d)
    y - Target vector (N x 1)
    lambd - Regularization parameter (scalar)

    Returns:
    error - Regularized squared loss (scalar)
    error_grad - Gradient of the loss with respect to w (flattened for minimize)
    """
    # Reshape w to be a column vector if necessary
    w = np.atleast_2d(w).T if w.ndim == 1 else w

    # Compute residuals
    residuals = y - X @ w  # Residuals (y - Xw)

    # Compute regularized loss
    error = 0.5 * (residuals.T @ residuals + lambd * w.T @ w).item()

    # Compute gradient
    error_grad = -(X.T @ residuals) + lambd * w

    # Return error and gradient (gradient must be flattened for minimize)
    return error, error_grad.flatten()

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


# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init.flatten(), jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()


# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(mses3)]
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()

import numpy as np
import pickle
import matplotlib.pyplot as plt
import sys

import numpy as np


def ldaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix

    # Find unique classes
    classes = np.unique(y)
    k = len(classes)  # Number of classes
    d = X.shape[1]  # Number of features

    # Initialize means matrix
    means = np.zeros((d, k))

    # Compute means for each class
    for idx, cls in enumerate(classes):
        X_cls = X[y.flatten() == cls]
        means[:, idx] = np.mean(X_cls, axis=0)

    # Compute common covariance matrix
    covmat = np.cov(X, rowvar=False, bias=True)

    return means, covmat


def qdaLearn(X, y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes

    # Find unique classes
    classes = np.unique(y)
    k = len(classes)  # Number of classes
    d = X.shape[1]  # Number of features

    # Initialize means matrix and covariance matrices list
    means = np.zeros((d, k))
    covmats = []

    # Compute means and covariance matrices for each class
    for idx, cls in enumerate(classes):
        X_cls = X[y.flatten() == cls]
        means[:, idx] = np.mean(X_cls, axis=0)
        covmat_cls = np.cov(X_cls, rowvar=False, bias=True)
        covmats.append(covmat_cls)

    return means, covmats


def ldaTest(means, covmat, Xtest, ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    #
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # Compute inverse of the covariance matrix
    inv_covmat = np.linalg.inv(covmat)

    # Precompute terms for discriminant function
    means_invcov = inv_covmat @ means  # d x k
    quadratic_terms = np.sum(means * means_invcov, axis=0)  # k x 1

    # Compute discriminant scores for each class
    linear_terms = Xtest @ means_invcov  # N x k
    g = linear_terms - 0.5 * quadratic_terms  # N x k

    # Predict classes based on highest score
    ypred_idx = np.argmax(g, axis=1)
    ypred = ypred_idx + 1  # Assuming classes are labeled from 1 to k
    ypred = ypred.reshape(-1, 1)

    # Calculate accuracy
    acc = 100 * np.mean(ypred.flatten() == ytest.flatten())

    return acc, ypred


def qdaTest(means, covmats, Xtest, ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    #
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    k = means.shape[1]  # Number of classes
    N = Xtest.shape[0]  # Number of test samples
    scores = np.zeros((N, k))

    for idx in range(k):
        mean_vec = means[:, idx]
        covmat = covmats[idx]
        inv_covmat = np.linalg.inv(covmat)
        sign, logdet = np.linalg.slogdet(covmat)

        diff = Xtest - mean_vec
        mdist = np.sum((diff @ inv_covmat) * diff, axis=1)

        # Compute discriminant function
        scores[:, idx] = -0.5 * (logdet + mdist)

    # Predict classes based on highest score
    ypred_idx = np.argmax(scores, axis=1)
    ypred = ypred_idx + 1  # Assuming classes are labeled from 1 to k
    ypred = ypred.reshape(-1, 1)

    # Calculate accuracy
    acc = 100 * np.mean(ypred.flatten() == ytest.flatten())

    return acc, ypred


# Load the sample data
if sys.version_info.major == 2:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'))
else:
    X, y, Xtest, ytest = pickle.load(open('sample.pickle', 'rb'), encoding='latin1')

# Train and test LDA
means_lda, covmat_lda = ldaLearn(X, y)
lda_acc, lda_preds = ldaTest(means_lda, covmat_lda, Xtest, ytest)
print(f"LDA Accuracy: {lda_acc:.2f}%")

# Train and test QDA
means_qda, covmats_qda = qdaLearn(X, y)
qda_acc, qda_preds = qdaTest(means_qda, covmats_qda, Xtest, ytest)
print(f"QDA Accuracy: {qda_acc:.2f}%")

# Plot decision boundaries
x1 = np.linspace(-5, 20, 100)
x2 = np.linspace(-5, 20, 100)
xx1, xx2 = np.meshgrid(x1, x2)
xx = np.zeros((x1.shape[0] * x2.shape[0], 2))
xx[:, 0] = xx1.ravel()
xx[:, 1] = xx2.ravel()

fig = plt.figure(figsize=[12, 6])

# LDA boundary
z_acc, z_lda_preds = ldaTest(means_lda, covmat_lda, xx, np.zeros((xx.shape[0], 1)))
plt.subplot(1, 2, 1)
plt.contourf(x1, x2, z_lda_preds.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest.flatten())
plt.title("LDA Decision Boundary")

# QDA boundary
z_acc, z_qda_preds = qdaTest(means_qda, covmats_qda, xx, np.zeros((xx.shape[0], 1)))
plt.subplot(1, 2, 2)
plt.contourf(x1, x2, z_qda_preds.reshape((x1.shape[0], x2.shape[0])), alpha=0.3)
plt.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest.flatten())
plt.title("QDA Decision Boundary")

plt.show()

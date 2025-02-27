import numpy as np
import pickle
import matplotlib.pyplot as plt
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
    discriminants = []
    for mean in means.T:
        diff = Xtest - mean
        discriminants.append(-0.5 * np.sum(diff @ covmat_inv * diff, axis=1))
    discriminants = np.array(discriminants).T

    # Predict class with the largest discriminant
    ypred = np.argmax(discriminants, axis=1) + 1  # Adding 1 for class labels
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

# Main script

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

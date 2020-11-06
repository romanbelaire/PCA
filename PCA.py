###AUTHOR: ROMAN BELAIRE
##12/19/19
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math

def normalize(X):
    """Normalize the input X using mean and standard variation method.
    Data d in X is normalized to equal [d - avg(x)] / standard deviation
    """
    m = X.shape[0]
    meanX = np.mean(X, axis=0)
    stdX = np.std(X, axis=0)

    X = (X - np.tile(meanX, (m,1)))/np.tile(stdX,(m,1))
    return X

def pca(X,k):
    """Compute PCA on k dimensions.
    This function takes input X, calculates covariance matrix and eigenvectors to get K principal components.
    Returns matrix X_pca which contains X projected onto K principal components.
    """
    m = X.shape[0]
    covariance = np.dot(X.T, X) / (m-1)
    # Eigen decomposition
    eigenvals, eigenvecs = np.linalg.eig(covariance)
    #project in K dimensions
    print("First two Principal Components from PCA:")
    print(eigenvecs[:,:k])
    pca_result = np.dot(X, eigenvecs[:,:k])
    return pca_result

def svd(X,k):
    """Perform PCA using SVD on k dimensions.
    Uses numpy.linalg.svd to compute principal components, then projects onto them in two dimensions.
    Returns resultant matrix.
    """
    unit, singular, unit2 = np.linalg.svd(X, full_matrices=False,compute_uv=True)
    print("First two Principal Components from SVD:")
    unit2 = np.linalg.inv(unit2)
    print(unit2[:,:k])
    svd = np.dot(X, unit2[:,:k])
    return svd

if __name__ == '__main__':
    k = 2   #dimensions to reduce to
    data = scipy.io.loadmat('cars.mat')
    X = data['X']
    X = np.delete(X, range(0,7),1)
    print("Original Data:")
    print(X,'\n\n')

    X = normalize(X)
    print("Normalized Data:")
    print(X, '\n\n')
    X_PCA = pca(X,k)
    X_SVD = svd(X,k)

    fig = plt.figure()
    plot = fig.add_subplot(1,1,1)

    plot.scatter(X[:,0], X[:,1], c='black', label='Original')
    plot.scatter(X_PCA[:,0], X_PCA[:,1], c='red', label='PCA')
    plot.scatter(X_SVD[:,0], X_SVD[:,1], c='blue', label='SVD')
    plot.legend()
    plt.title("Projecting onto PCs in 2D")
    plt.show()

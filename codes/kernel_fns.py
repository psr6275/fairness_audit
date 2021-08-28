import numpy as np

def rbf_kernel(X, Y=None, gamma=None):
    
    if gamma in [None,'scale']:
        gamma = 1.0 / X.shape[1]

    if Y is None:
        Y = X
    XX = np.einsum('ij,ij->i', X, X)[:,np.newaxis]
    YY = np.einsum('ij,ij->i', Y, Y)[:,np.newaxis]
    XY = np.dot(X,Y.T)
    K = -2*XY
    K += XX
    K += YY
#     K = euclidean_distances(X, Y, squared=True)
    np.maximum(K,0,out=K)
    if X is Y:
        np.fill_diagonal(K,0)
    K *= -gamma
    np.exp(K, K)  # exponentiate K in-place
    return K

def linear_kernel(x,y):
    if y is None:
        y = x
    return np.dot(x,y.T)
def poly_kernel(X, Y=None, degree=3, gamma=None, coef0=1):
    """
    Compute the polynomial kernel between X and Y::
        K(X, Y) = (gamma <X, Y> + coef0)^degree
    Read more in the :ref:`User Guide <polynomial_kernel>`.
    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features), default=None
    degree : int, default=3
    gamma : float, default=None
        If None, defaults to 1.0 / n_features.
    coef0 : float, default=1
    Returns
    -------
    Gram matrix : ndarray of shape (n_samples_X, n_samples_Y)
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    K = np.dot(X, Y.T)
    K *= gamma
    K += coef0
    K **= degree
    return K
def sigmoid_kernel(X, Y=None, gamma=None, coef0=1):
    """
    Compute the sigmoid kernel between X and Y::
        K(X, Y) = tanh(gamma <X, Y> + coef0)
    Read more in the :ref:`User Guide <sigmoid_kernel>`.
    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features), default=None
    gamma : float, default=None
        If None, defaults to 1.0 / n_features.
    coef0 : float, default=1
    Returns
    -------
    Gram matrix : ndarray of shape (n_samples_X, n_samples_Y)
    """
    if gamma in [None,'scale']:
        gamma = 1.0 / X.shape[1]

    K = np.dot(X, Y.T)
    K *= gamma
    K += coef0
    np.tanh(K, K)  # compute tanh in-place
    return K

def calculate_kernel(X,Y=None, kernel = 'linear', **kwds):
    assert kernel in ['linear', 'rbf','poly','sigmoid']
    metric = eval(kernel+'_kernel')
    out = metric(X,Y,**kwds)
    return out


"""Utility functions for the Poor Man's GPLVM."""

import numpy as np


def rbf_kernel(X, Y=None, length_scale=1.0):
    """Radial Basis Function kernel.
    
    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        Left argument of the kernel
    Y : array-like of shape (n_samples_Y, n_features), default=None
        Right argument of the kernel. If None, Y=X.
    length_scale : float, default=1.0
        Length scale parameter
        
    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        Kernel matrix
    """
    X = np.asarray(X)
    if Y is None:
        Y = X
    else:
        Y = np.asarray(Y)
        
    XX = np.sum(X**2, axis=1)[:, np.newaxis]
    YY = np.sum(Y**2, axis=1)[np.newaxis, :]
    XY = np.dot(X, Y.T)
    
    # Compute squared euclidean distance
    sq_dists = XX + YY - 2 * XY
    
    # Apply RBF kernel
    K = np.exp(-0.5 * sq_dists / (length_scale**2))
    
    return K


def pca_init(Y, latent_dim):
    """Initialize latent points using PCA.
    
    Parameters
    ----------
    Y : array-like of shape (n_samples, n_features)
        Observed data
    latent_dim : int
        Dimensionality of the latent space
        
    Returns
    -------
    X : ndarray of shape (n_samples, latent_dim)
        Initial latent points
    """
    Y = np.asarray(Y)
    n_samples = Y.shape[0]
    
    # Center the data
    Y_centered = Y - np.mean(Y, axis=0)
    
    # Compute SVD
    U, S, Vh = np.linalg.svd(Y_centered, full_matrices=False)
    
    # Return the first latent_dim principal components
    X = U[:, :latent_dim] * S[:latent_dim]
    
    return X 
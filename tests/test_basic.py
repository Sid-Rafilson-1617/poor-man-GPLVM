"""Basic tests for poor_man_gplvm."""

import numpy as np
import pytest

from poor_man_gplvm.core import GPLVM
from poor_man_gplvm.utils import rbf_kernel, pca_init


def test_rbf_kernel():
    """Test the RBF kernel function."""
    X = np.array([[0, 0], [1, 1]])
    K = rbf_kernel(X)
    
    # Test shape
    assert K.shape == (2, 2)
    
    # Test diagonal is 1
    assert np.allclose(np.diag(K), 1.0)
    
    # Test symmetry
    assert np.allclose(K, K.T)


def test_pca_init():
    """Test PCA initialization."""
    Y = np.random.randn(10, 5)
    X = pca_init(Y, latent_dim=2)
    
    # Test shape
    assert X.shape == (10, 2)


def test_gplvm_init():
    """Test GPLVM initialization."""
    model = GPLVM(latent_dim=3)
    
    assert model.latent_dim == 3
    assert model.X is None
    assert model.Y is None


def test_gplvm_fit():
    """Test GPLVM fit method."""
    Y = np.random.randn(10, 5)
    model = GPLVM(latent_dim=2)
    model.fit(Y)
    
    assert model.Y.shape == (10, 5)
    assert model.X.shape == (10, 2)


def test_gplvm_transform():
    """Test GPLVM transform method."""
    Y_train = np.random.randn(10, 5)
    Y_test = np.random.randn(3, 5)
    
    model = GPLVM(latent_dim=2)
    model.fit(Y_train)
    X_test = model.transform(Y_test)
    
    assert X_test.shape == (3, 2) 
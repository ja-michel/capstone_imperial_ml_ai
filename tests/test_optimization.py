
import pytest
import numpy as np
from ml.gp_optimizer import GPOptimizer

def test_parse_input():
    optimizer = GPOptimizer()
    res = optimizer.parse_input("0.1-0.2")
    np.testing.assert_array_equal(res, np.array([0.1, 0.2]))
    
    with pytest.raises(ValueError):
        optimizer.parse_input("invalid")
    
    with pytest.raises(ValueError):
        optimizer.parse_input("0.1-0.2-0.3")

def test_gp_fit_predict():
    optimizer = GPOptimizer()
    X = np.array([[0.1, 0.1], [0.9, 0.9]])
    y = np.array([1.0, 2.0])
    
    optimizer.fit(X, y)
    
    # Predict at training points should be close to actual values (assuming low noise in kernel or sufficient fit)
    mu, std = optimizer.predict(X, return_std=True)
    np.testing.assert_allclose(mu, y, atol=1e-1) # large tolerance due to noise kernel
    
    # Predict new point
    mu_new = optimizer.predict(np.array([[0.5, 0.5]]))
    assert mu_new.shape == (1,)

def test_expected_improvement():
    optimizer = GPOptimizer()
    X = np.array([[0.1, 0.1], [0.9, 0.9]])
    y = np.array([1.0, 2.0]) # Max is 2.0
    optimizer.fit(X, y)
    
    # EI at known max should be low (unless xi is large or noise is high), 
    # but at unknown area (0.5, 0.5) it should be non-zero probably.
    ei = optimizer.expected_improvement(np.array([[0.5, 0.5]]), xi=0.01)
    assert ei >= 0.0

def test_suggest_next_point():
    optimizer = GPOptimizer()
    X = np.array([[0.1, 0.1]])
    y = np.array([1.0])
    optimizer.fit(X, y)
    
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    next_point = optimizer.suggest_next_point(bounds)
    
    assert next_point.shape == (2,)
    assert 0.0 <= next_point[0] <= 1.0
    assert 0.0 <= next_point[1] <= 1.0

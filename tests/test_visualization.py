
import pytest
import numpy as np
from ml.gp_optimizer import GPOptimizer
from ml.visualization import plot_gp_search_space
import matplotlib.pyplot as plt

def test_plot_gp_search_space(monkeypatch):
    # Mock plt.show to prevent blocking or GUI errors
    monkeypatch.setattr(plt, "show", lambda: None)
    
    optimizer = GPOptimizer()
    X = np.array([[0.1, 0.1], [0.9, 0.9]])
    y = np.array([1.0, 2.0])
    optimizer.fit(X, y)
    
    bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
    
    # Should run without error
    plot_gp_search_space(optimizer, bounds)

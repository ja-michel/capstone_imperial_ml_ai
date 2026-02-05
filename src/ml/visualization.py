
import numpy as np
import matplotlib.pyplot as plt
from ml.models.model import GPOptimizer

def plot_gp_search_space(optimizer: GPOptimizer, bounds: np.ndarray, xi: float = 0.01):
    """
    Plots the GP mean, Uncertainty (Std Dev), and Expected Improvement.
    
    Args:
        optimizer: Fitted GPOptimizer instance.
        bounds: Bounds of the search space (2, 2).
        xi: Exploration-exploitation trade-off.
    """
    if optimizer.X_train is None:
        print("No model fitted yet. Cannot plot.")
        return

    # Create a grid
    res = 50
    x1 = np.linspace(bounds[0, 0], bounds[0, 1], res)
    x2 = np.linspace(bounds[1, 0], bounds[1, 1], res)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.array([X1.ravel(), X2.ravel()]).T

    # Predict
    mu, std = optimizer.predict(X_grid, return_std=True)
    ei = optimizer.expected_improvement(X_grid, xi=xi)

    mu = mu.reshape(X1.shape)
    std = std.reshape(X1.shape)
    ei = ei.reshape(X1.shape)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot Mean
    c0 = axes[0].contourf(X1, X2, mu, levels=20, cmap='viridis')
    axes[0].scatter(optimizer.X_train[:, 0], optimizer.X_train[:, 1], c='r', marker='x', label='Observed')
    axes[0].set_title("GP Predicted Mean")
    fig.colorbar(c0, ax=axes[0])
    axes[0].legend()

    # Plot Uncertainty
    c1 = axes[1].contourf(X1, X2, std, levels=20, cmap='plasma')
    axes[1].scatter(optimizer.X_train[:, 0], optimizer.X_train[:, 1], c='r', marker='x')
    axes[1].set_title("Uncertainty (Std Dev)")
    fig.colorbar(c1, ax=axes[1])

    # Plot Expected Improvement
    c2 = axes[2].contourf(X1, X2, ei, levels=20, cmap='inferno')
    axes[2].scatter(optimizer.X_train[:, 0], optimizer.X_train[:, 1], c='r', marker='x')
    axes[2].set_title(f"Expected Improvement (xi={xi})")
    fig.colorbar(c2, ax=axes[2])
    
    # Suggest next point visual marker
    # We could optimize here just to show it, or pass it in. 
    # For now, let's just leave it as the heatmap showing the "hot" areas.

    plt.tight_layout()
    plt.show()

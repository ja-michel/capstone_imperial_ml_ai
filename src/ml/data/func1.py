from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.pyplot import plot, ion, show, draw
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent

data_file_path = PROJECT_DIR / "data" / "f1_data.csv"

if not data_file_path.exists():
    raise FileNotFoundError(f"File not found: {data_file_path}")


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

def load_data(file_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV file.
    
    Returns:
        Tuple of (X, y) where X is the input data and y is the target data.
    """
    print(f"Loading data from {file_path}")
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    data = pd.read_csv(file_path)
    if data.empty:
        raise ValueError("Data file is empty.")
    return data[[c for c in data.columns if c.startswith("x")]].values, data["target"].values


X, y = load_data(data_file_path)

X = X[:, :-1]

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


def transform_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_log = np.log(np.abs(X))
    y_log = np.log(np.abs(y)).reshape(-1, 1)
    x_scaler = StandardScaler(with_mean=True, with_std=True)
    y_scaler = RobustScaler()
    X_transformed = x_scaler.fit_transform(x_log)
    y_transformed = y_scaler.fit_transform(y_log).ravel()
    print(f"X_transformed shape: {X_transformed.shape}")
    print(f"y_transformed shape: {y_transformed.shape}")
    return X_transformed, y_transformed

X_transformed, y_transformed = transform_data(X, y)

from ml.plot import PlotUtils

plotter = PlotUtils()
plotter.scatter(X, y, title="Original Data")
plotter.scatter(X_transformed, y_transformed, title="Transformed Data")
plotter.show()



from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from ml.models import GPOptimizer

def get_bounds(X_final):
    """
    Calculate safe bounds for the optimization based on the empirical data.
    Adds a buffer to allow exploration outside the current sampled region.
    """
    # Assuming X_final is your scaled input array (N_samples, N_features)

    # 1. Calculate the empirical min and max for each feature (dimension)
    x_min = np.min(X_final, axis=0)
    x_max = np.max(X_final, axis=0)

    # 2. Add a small buffer (e.g., 25%) for exploration outside the current sampled region
    buffer = 0.25 # Adds 25% extra room on either side

    # 3. Create the final bounds list of tuples
    bounds_list = []
    for i in range(X_final.shape[1]): # Iterate through each feature
        # Calculate the range and buffer size
        feature_range = x_max[i] - x_min[i]
        buffer_size = feature_range * buffer
        
        # Define the new bounds (min - buffer, max + buffer)
        new_min = x_min[i] - buffer_size
        new_max = x_max[i] + buffer_size
        
        bounds_list.append((new_min, new_max))

    # bounds_list is now the correct variable to pass to propose_next_location
    print("Calculated Safe Bounds:", bounds_list)

    return bounds_list

bounds = get_bounds(X_transformed)
# kernel = Matern(length_scale=[1.0, 1.0], length_scale_bounds=[(1e-2, 10.0), (1e-2, 10.0)], nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))
kernel = (
    ConstantKernel(.001, (1e-15, 10)) *
    Matern(length_scale=[.2], length_scale_bounds=(1e-15, 10), nu=2.5) +
    WhiteKernel(noise_level=1e-10, noise_level_bounds=(1e-20, 10))
)
gp = GPOptimizer(kernel=kernel)
gp.fit(X_transformed, y_transformed)

import matplotlib.pyplot as plt
# --- 2. The Plotting Function ---
def plot_bayesian_optimization(model: GPOptimizer, X_train: np.ndarray, y_train: np.ndarray, bounds: list[tuple[float, float]], scaler_y=None):
    """
    Plots the GP model (Top) and Acquisition Function (Bottom).
    
    Args:
        model: The trained GaussianProcessRegressor
        X_train: The raw input data (scaled internally for prediction if needed)
        y_train: The raw output data (log-transformed internally)
        bounds: The limits of the search space [(min, max), ...]
        scaler_y: The scaler used on y (optional, for un-scaling plots)
    """
    
    # A. Create a grid of points to plot the smooth lines
    x_grid = np.linspace(bounds[0][0], bounds[0][1], 1000).reshape(-1, 1)
    
    # B. Get Model Predictions (Mean & Uncertainty) on the grid
    # Note: These predictions are in "Transformed Space" (Log-Scaled)
    mu, sigma = model.predict(x_grid, return_std=True)
    
    # C. Calculate Acquisition Function (EI) on the grid
    # We look for improvement over the best CURRENT observed value
    y_best_current = np.max(model.y_train_) 
    ei = model.expected_improvement(x_grid, y_best_current)
    
    # --- D. PLOTTING ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Top Plot: The Surrogate Model
    ax1 = axes[0]
    ax1.plot(x_grid, mu, 'b-', label='GP Mean')
    ax1.fill_between(x_grid.ravel(), 
                     mu - 1.96 * sigma, 
                     mu + 1.96 * sigma, 
                     alpha=0.2, 
                     color='blue', 
                     label='95% Confidence')
    
    # Plot the actual data points (training data)
    ax1.scatter(X_train, y_train, c='red', s=50, zorder=10, label='Observations')
    
    ax1.set_title("Gaussian Process Surrogate Model")
    ax1.set_ylabel("Target Value (Transformed)")
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Bottom Plot: The Acquisition Function
    ax2 = axes[1]
    ax2.plot(x_grid, ei, 'g-', linewidth=2, label='Expected Improvement (EI)')
    ax2.fill_between(x_grid.ravel(), ei, 0, color='green', alpha=0.1)
    
    # Highlight the next best point
    next_x_idx = np.argmax(ei)
    next_x = x_grid[next_x_idx]
    ax2.axvline(next_x, color='k', linestyle='--', label=f'Next Sample: {next_x[0]:.3f}')
    
    ax2.set_title("Acquisition Function")
    ax2.set_xlabel("Input Space (x)")
    ax2.set_ylabel("EI Score")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_learned_kernel(model):
    """
    Plots the Correlation vs. Distance for each feature in the learned kernel.
    """
    # 1. Extract the learned kernel (The Matern part)
    # Note: model.kernel_ is usually (Matern + WhiteKernel). 
    # We need to find the Matern part to get the length scales.
    
    learned_kernel = model.kernel_
    
    # If it's a Sum (Matern + Noise), getting the Matern part:
    # if hasattr(learned_kernel, 'k1') and isinstance(learned_kernel.k1, Matern):
    #     matern_part = learned_kernel.k1
    # elif hasattr(learned_kernel, 'k2') and isinstance(learned_kernel.k2, Matern):
    #     matern_part = learned_kernel.k2
    # else:
    #     # Fallback: assume the whole thing is Matern if no noise kernel was added
    #     matern_part = learned_kernel

    def find_matern(kernel):
        if isinstance(kernel, Matern):
            return kernel
        if hasattr(kernel, 'k1'):
            return find_matern(kernel.k1) or find_matern(kernel.k2)
        return None

    matern_part = find_matern(learned_kernel) or learned_kernel


    length_scales = matern_part.length_scale
    nu = matern_part.nu
    
    print(f"Learned Length Scales: {length_scales}")
    print(f"Smoothness (nu): {nu}")

    # 2. Setup the Plot
    # We plot distance from 0 to 3x the max length scale to see the full decay
    max_ls = np.max(length_scales)
    d = np.linspace(0, max_ls * 3, 100)
    
    plt.figure(figsize=(10, 6))
    
    # 3. Calculate and Plot Decay for each Feature
    # Formula for Matern 5/2 correlation:
    # (1 + sqrt(5)*d/L + 5*d^2/(3*L^2)) * exp(-sqrt(5)*d/L)
    
    colors = ['#1f77b4', '#ff7f0e'] # Blue, Orange
    if not isinstance(length_scales, (list, tuple)):
        length_scales = [length_scales]
    # if len(length_scales) == 1:
    #     length_scales = (length_scales,)

    for i, L in enumerate(length_scales):
        # We manually compute the correlation vector for visualization
        # This is strictly the Matern correlation formula
        sqrt_5 = np.sqrt(5)
        scaled_d = d / L
        correlation = (1 + sqrt_5 * scaled_d + (5/3) * (scaled_d**2)) * np.exp(-sqrt_5 * scaled_d)
        
        plt.plot(d, correlation, lw=3, color=colors[i % 2], label=f'Feature {i+1} (L={L:.3f})')
        
        # Mark the point where correlation drops to ~0.1 (effectively "uncorrelated")
        cutoff_idx = np.searchsorted(-correlation, -0.1) # efficient search
        if cutoff_idx < len(d):
            plt.scatter(d[cutoff_idx], correlation[cutoff_idx], color=colors[i % 2], zorder=5)

    # 4. Styling
    plt.title(f"Learned Kernel Physics (Matérn {nu})", fontsize=14)
    plt.xlabel("Distance between points (Normalized Space)", fontsize=12)
    plt.ylabel("Correlation (Similarity)", fontsize=12)
    plt.axhline(0, color='black', alpha=0.3, linestyle='--')
    plt.axhline(1.0, color='black', alpha=0.3, linestyle='--')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.show()

plot_bayesian_optimization(gp, X_transformed, y_transformed, bounds)
# visualize_learned_kernel(gp.gp)
# import time
# import threading
# def callback():
#     print("Script running. Press Ctrl+C to exit.")

# timer = threading.Timer(60.0, callback)
# timer.start() 

try:
    input("Press Enter to exit...")
except KeyboardInterrupt:
    print("\nExiting script...")

# import numpy as np
# from matplotlib import pyplot as plt


# def main():
#     plt.axis([-10, 10, -10, 10])  # Adjust axis limits for various functions
#     plt.ion()
#     plt.show()

#     x = np.arange(-10, 11)

#     # Plot different mathematical functions
#     for function_name in ["Linear", "Quadratic", "Cubic", "Square Root"]:
#         if function_name == "Linear":
#             y = x
#         elif function_name == "Quadratic":
#             y = x**2
#         elif function_name == "Cubic":
#             y = x**3
#         elif function_name == "Square Root":
#             y = np.sqrt(np.abs(x))

#         plt.plot(x, y, label=function_name)
#         plt.draw()
#         plt.pause(0.001)
#         input("Press [enter] to continue.")

# if __name__ == '__main__':
#     main()
from pathlib import Path
import numpy as np
import pandas as pd
import sys
from dataclasses import dataclass
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from ml.models import GPOptimizer
+
@dataclass
class OptimiserFunc1:
    X: np.ndarray
    y: np.ndarray
    X_transformed: np.ndarray | None
    y_transformed: np.ndarray | None
    
    @staticmethod
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


    def __post_init__(self):
        sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
        PROJECT_DIR = Path(__file__).resolve().parent.parent.parent.parent
        data_file_path = PROJECT_DIR / "data" / "f1_data.csv"
        if not data_file_path.exists():
            raise FileNotFoundError(f"File not found: {data_file_path}")
        
        self.X, self.y = OptimiserFunc1.load_data(data_file_path)
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")
        self.X_transformed, self.y_transformed = OptimiserFunc1.transform(self.X, self.y)

    def plot(self):
        from ml.plot import PlotUtils
        plotter = PlotUtils()
        plotter.scatter(self.X, self.y, title="Original Data")
        if self.X_transformed is not None and self.y_transformed is not None:
            plotter.scatter(self.X_transformed, self.y_transformed, title="Transformed Data")
        plotter.show()

    @staticmethod
    def transform(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_log = np.log(X)
        y_log = np.log(y).reshape(-1, 1)
        x_scaler = StandardScaler(with_mean=True, with_std=True)
        y_scaler = RobustScaler()
        X_transformed = x_scaler.fit_transform(x_log)
        y_transformed = y_scaler.fit_transform(y_log).ravel()
        print(f"X_transformed shape: {X_transformed.shape}")
        print(f"y_transformed shape: {y_transformed.shape}")
        return X_transformed, y_transformed
    
    def inverse_transform(self, X_transformed: np.ndarray, y_transformed: np.ndarray):
        x_scaler = StandardScaler(with_mean=True, with_std=True)
        y_scaler = RobustScaler()
        X = x_scaler.inverse_transform(X_transformed)
        y = y_scaler.inverse_transform(y_transformed).ravel()
        return X, y 
    
    def optimise(self):
        best_ls = OptimiserFunc1.find_best_length_scale_multidim(self.X_transformed, self.y_transformed)
        recommended = OptimiserFunc1.recommend_bounds(X_transformed)
        print(f"Recommended Bounds: {recommended}")
        pass
    
    def recommend_bounds(X):
        # Calculate pairwise distances between all points
        dists = pdist(X, metric='euclidean')
        
        # 1. Lower Bound: Based on the minimum meaningful resolution
        # We take the 10th percentile of distances to avoid outliers
        # If points are identical, dist is 0, so we filter those out first
        non_zero_dists = dists[dists > 1e-5] 
        if len(non_zero_dists) > 0:
            min_bound = np.percentile(non_zero_dists, 10) 
        else:
            min_bound = 1e-2 # Fallback
            
        # 2. Upper Bound: The diameter of the whole space
        max_bound = np.max(dists) * 1.5 # Add 50% buffer
        
        return (min_bound, max_bound)


    @staticmethod
    def find_best_length_scale_multidim(X, y, search_space_min: float = 1e-2, search_space_max: float = 1e2, n_iter=50):
        """
        Randomly searches for the best length scale vector (ARD) 
        that minimizes Leave-One-Out Cross-Validation error.
        """
        n_features = X.shape[1]
        
        # 1. Define Search Space (Log-Uniform)
        # We search between 0.1 and 10.0 (adjust based on your scaling)
        log_min, log_max = np.log(search_space_min), np.log(search_space_max)
        
        best_score = float('inf')
        best_ls_vector = None
        
        results = [] # To store (length_scale_vector, mse) for plotting
        
        print(f"Starting Random Search with {n_iter} iterations on {n_features} dimensions...")
        
        # Use scikit-learn's cross-validator for cleaner code
        loo = LeaveOneOut()
        
        for i in range(n_iter):
            # 2. Sample a random length scale vector (one for each dimension)
            # We use log-uniform sampling so we explore small values (0.001) as much as large (5)
            random_ls = np.exp(np.random.uniform(log_min, log_max, size=n_features))
            
            # 3. Create Model with FIXED parameters (optimizer=None)
            # We want to test *this exact* length scale, not let the GP change it.
            kernel = Matern(length_scale=random_ls, length_scale_bounds="fixed", nu=2.5)
            model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None)
            
            # 4. Calculate Cross-Validation Error (MSE)
            # validation scores are negative MSE in sklearn, so we negate them back
            scores = cross_val_score(model, X, y, cv=loo, scoring='neg_mean_squared_error')
            mse = -np.mean(scores)
            
            results.append((random_ls, mse))
            
            if mse < best_score:
                best_score = mse
                best_ls_vector = random_ls
                
        print(f"\nBest MSE: {best_score:.4f}")
        print(f"Best Length Scales: {np.round(best_ls_vector, 3)}")
        
        return best_ls_vector, results










# def transform_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     x_log = np.log(np.abs(X))
#     y_log = np.log(np.abs(y)).reshape(-1, 1)
#     x_scaler = StandardScaler(with_mean=True, with_std=True)
#     y_scaler = RobustScaler()
#     X_transformed = x_scaler.fit_transform(x_log)
#     y_transformed = y_scaler.fit_transform(y_log).ravel()
#     print(f"X_transformed shape: {X_transformed.shape}")
#     print(f"y_transformed shape: {y_transformed.shape}")
#     return X_transformed, y_transformed

# X_transformed, y_transformed = transform_data(X, y)






# def get_bounds(X_final):
#     """
#     Calculate safe bounds for the optimization based on the empirical data.
#     Adds a buffer to allow exploration outside the current sampled region.
#     """
#     # Assuming X_final is your scaled input array (N_samples, N_features)

#     # 1. Calculate the empirical min and max for each feature (dimension)
#     x_min = np.min(X_final, axis=0)
#     x_max = np.max(X_final, axis=0)

#     # 2. Add a small buffer (e.g., 25%) for exploration outside the current sampled region
#     buffer = 0.25 # Adds 25% extra room on either side

#     # 3. Create the final bounds list of tuples
#     bounds_list = []
#     for i in range(X_final.shape[1]): # Iterate through each feature
#         # Calculate the range and buffer size
#         feature_range = x_max[i] - x_min[i]
#         buffer_size = feature_range * buffer
        
#         # Define the new bounds (min - buffer, max + buffer)
#         new_min = x_min[i] - buffer_size
#         new_max = x_max[i] + buffer_size
        
#         bounds_list.append((new_min, new_max))

#     # bounds_list is now the correct variable to pass to propose_next_location
#     print("Calculated Safe Bounds:", bounds_list)

#     return bounds_list

# bounds = get_bounds(X_transformed)
# kernel = Matern(length_scale=[1.0, 1.0], length_scale_bounds=[(1e-2, 10.0), (1e-2, 10.0)], nu=2.5) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e1))

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error

from scipy.spatial.distance import pdist, squareform
import numpy as np



# def find_best_length_scale_1d(X, y):
#     # Define a grid of length scales to test (e.g., 0.1 to 10.0)
#     grid = np.logspace(np.log10(0.001), np.log10(5.0), 1000)
#     scores = []
    
#     loo = LeaveOneOut()
    
#     for ls in grid:
#         errors = []
#         # We fix the length_scale and prevent the optimizer from changing it
#         # by setting bounds extremely tight around the value
#         # kernel = Matern(length_scale=ls, length_scale_bounds="fixed", nu=2.5) 
#         kernel = (
#             # ConstantKernel(.001, (1e-15, 10)) *
#             Matern(length_scale=ls, length_scale_bounds=recommended, nu=2.5) +
#             WhiteKernel(noise_level=1e-5, noise_level_bounds="fixed")
#         )
#         model = GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None)
        
#         for train_idx, test_idx in loo.split(X):
#             X_t, X_v = X[train_idx], X[test_idx]
#             y_t, y_v = y[train_idx], y[test_idx]
            
#             model.fit(X_t, y_t)
#             y_pred = model.predict(X_v)
#             errors.append((y_v - y_pred)**2)
            
#         scores.append(np.mean(errors))
        
#     best_ls = grid[np.argmin(scores)]
#     print(f"Best Length Scale by Cross-Validation: {best_ls}")
#     import matplotlib.pyplot as plt
#     # Plotting the Error Curve
#     plt.plot(grid, scores, marker='o')
#     plt.xscale('log')
#     plt.xlabel('Length Scale')
#     plt.ylabel('LOOCV Error (MSE)')
#     plt.title('Error vs Length Scale')
#     plt.show()
    
#     return best_ls

# best_ls = find_best_length_scale_1d(X_transformed, y_transformed)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.model_selection import LeaveOneOut, cross_val_score



# --- Example Usage ---
# Assuming X_train is your scaled 4-column data
best_ls, all_results = find_best_length_scale_multidim(X_transformed, y_transformed, n_iter=100)

kernel = (
    ConstantKernel(.001, (1e-15, 10)) *
    Matern(length_scale=best_ls, length_scale_bounds="fixed", nu=2.5) +
    WhiteKernel(noise_level=1e-10, noise_level_bounds="fixed")
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
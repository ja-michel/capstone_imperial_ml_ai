
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel
from scipy.stats import norm
from scipy.optimize import minimize
import typing

class GPOptimizer:
    def __init__(self, kernel=None, alpha=1e-10, n_restarts_optimizer=5):
        if kernel is None:
            # Standard kernel: Constant * Matern(nu=2.5) + Noise
            self.kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=[0.1, 0.1], length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level_bounds=(1e-10, 1e1))
        else:
            self.kernel = kernel
            
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel, 
            alpha=alpha, 
            n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=True
        )
        self.X_train = None
        self.y_train = None
    
    @staticmethod
    def parse_input(input_str: str) -> np.ndarray:
        """Parses 'input1-input2' string into a numpy array."""
        try:
            parts = input_str.split('-')
            if len(parts) != 2:
                raise ValueError
            return np.array([float(parts[0]), float(parts[1])])
        except (ValueError, IndexError):
            raise ValueError(f"Invalid input format: '{input_str}'. Expected format 'float-float' (e.g. '0.1-0.2').")

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the GP model to the data."""
        self.X_train = np.atleast_2d(X)
        self.y_train = np.atleast_1d(y)
        self.gp.fit(self.X_train, self.y_train)

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Wrapper for GP prediction."""
        return self.gp.predict(X, return_std=return_std)

    def expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """
        Computes the Expected Improvement at points X.
        
        Args:
            X: Points to evaluate EI at. Shape (n_samples, n_features).
            xi: Exploration-exploitation trade-off parameter. Higher values favor exploration.
            
        Returns:
            EI values. Shape (n_samples,).
        """
        X = np.atleast_2d(X)
        mu, sigma = self.gp.predict(X, return_std=True)
        
        if self.y_train is None:
            return np.zeros(X.shape[0])
            
        mu_sample_opt = np.max(self.y_train) if self.y_train is not None else 0.0
        # In this problem, we assumed the user gives outputs. 
        # Usually standard is maximization. If minimization, flip signs.
        # Assuming user provides function values where higher is better? 
        # The prompt didn't specify, but "optimization" usually implies maximization or finding an extrema.
        # Let's assume Maximization for EI. If user wants Min, they can flip sign of y.
        
        mu_sample_opt = np.max(self.y_train)

        with np.errstate(divide='warn'):
            imp = mu - mu_sample_opt - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

    def suggest_next_point(self, bounds: list, n_restarts: int = 10, xi: float = 0.01) -> np.ndarray:
        """
        Suggests the next point to query by maximizing Expected Improvement.
        """
        dim = bounds.shape[0]
        min_val = 1
        min_x = None
        
        def min_obj(X):
            # Minimization of negative expected improvement
            return -self.expected_improvement(X.reshape(1, -1), xi=xi)[0]

        # Find the best optimum by starting from n_restarts random points.
        for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
            res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < min_val:
                min_val = res.fun
                min_x = res.x           
                
        return min_x.reshape(-1)

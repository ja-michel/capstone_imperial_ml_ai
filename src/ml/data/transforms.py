

import numpy as np
from scipy.stats import zscore, pearsonr


class TransformUtils:
    @staticmethod
    def get_outliers(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """Identify outliers in the data using Z-score method. Return the indexes of outliers."""
        z_scores = zscore(data)
        outliers = np.where(np.abs(z_scores) >= threshold)[0]
        return data[outliers]

    @staticmethod
    def get_quantile_outliers(data: np.ndarray, lower_quantile: float = 0.25, upper_quantile: float = 0.75, factor: float = 1.5) -> np.ndarray:
        """Identify outliers in the data using the IQR method. Return the indexes of outliers."""
        Q1 = np.quantile(data, lower_quantile)
        Q3 = np.quantile(data, upper_quantile)
        IQR = Q3 - Q1
        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)
        outliers = np.where((data < lower_bound) | (data > upper_bound))[0]
        return outliers
    
    @staticmethod
    def get_corr_matrix(X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute the correlation matrix between X and y."""
        combined = np.column_stack((X, y))
        corr_matrix = np.corrcoef(combined, rowvar=False)
        return corr_matrix

    def pearsonsr(self, x: np.ndarray, y: np.ndarray, threshold: float = 0.7) -> float:
        """Compute Pearson's r between two arrays."""
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("Input arrays must be one-dimensional.")
        if x.shape[0] != y.shape[0]:
            raise ValueError("Input arrays must have the same length.")

        # Calculate Pearson's r for each pair of columns in X
        r_01, _ = pearsonr(x, y)
        correlated = (r_01 > threshold) or (r_01 < -threshold)
        print(f"Pearson's r between x and y: {r_01}"
            f" => correlated: {correlated}")
        return r_01
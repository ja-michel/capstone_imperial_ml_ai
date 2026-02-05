import numpy as np
import pandas as pd
import os
import sys
from ml.models.model import GPOptimizer
from ml.visualization import plot_gp_search_space
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel as C, WhiteKernel

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = os.path.join(PROJECT_DIR, "data")
BOUNDS = np.array([[0.0, 0.999999], [0.0, 0.999999]])

def load_data(func: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load data from CSV file.
    
    Returns:
        Tuple of (X, y) where X is the input data and y is the target data.
    """
    file_path = os.path.join(DATA_FILE, f"{func}_data.csv")
    print(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    data = pd.read_csv(file_path)
    if data.empty:
        raise ValueError("Data file is empty.")
    return data[[c for c in data.columns if c.startswith("x")]].values, data["target"].values

def save_data(df: pd.DataFrame):
    df.to_csv(DATA_FILE, index=False)


def main(func: str):
    print("Gaussian Process Optimization Explorer")
    print("--------------------------------------")

    # 1. Load Data
    X, y = load_data(func)
    print(f"Loaded {len(X)} observations.")

    kernel = Matern(length_scale=[1, 1], length_scale_bounds=(1e-2, 1e2), nu=2.5) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e1))
    optimizer = GPOptimizer(kernel=kernel)

    while True:
        if X is not None and y is not None:
            # 2. Fit Model
            print("Fitting GP model...")
            optimizer.fit(X, y)
            
            # 3. Optimize / Suggest Next Point
            print("Optimizing acquisition function (Expected Improvement)...")
            next_point = optimizer.suggest_next_point(BOUNDS)
            print(f"\n>>> SUGGESTED NEXT POINT: {next_point[0]:.6f}-{next_point[1]:.6f}")
            
            # 4. Visualize

            if X.shape[1] == 2:
                print("Generating plots...")
                plot_gp_search_space(optimizer, BOUNDS)


            print("Generating plots...")
            plot_gp_search_space(optimizer, BOUNDS)
        else:
            print("No data available. Suggesting random starting point.")
            next_point = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
            print(f"\n>>> SUGGESTED STARTING POINT: {next_point[0]:.6f}-{next_point[1]:.6f}")

        # 5. Interactive Input (Optional, for easy data entry)
        response = input("\nNew observation? [y/N]: ") or 'n'
        response = response.strip().lower()
        if response == 'n':
            break

        if response == 'y':
            input_str = input("Enter input (format float-float, e.g. 0.5-0.5): ").strip()
            try:
                x_vals = GPOptimizer.parse_input(input_str)
                output_val = float(input("Enter output value: ").strip())
                X = np.append(X, [x_vals], axis=0)
                y = np.append(y, output_val)
                print("Data saved. Rerun the script to update model and suggestion.")
            except ValueError as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="GP Optimization Script")
    parser.add_argument("function", type=str, help="Name of the function to optimize", choices=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8"])
    args = parser.parse_args()
    print(f"Optimizing function: {args.function}")
    try:
        main(func=args.function)
    except KeyboardInterrupt:
        print("\n\nUser interrupted the program. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")
        raise

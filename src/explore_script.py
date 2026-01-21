import numpy as np
import pandas as pd
import os
import sys
from ml.optimization import GPOptimizer
from ml.visualization import plot_gp_search_space

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_FILE = os.path.join(PROJECT_DIR, "data", "func2.csv")
BOUNDS = np.array([[0.0, 0.999999], [0.0, 0.999999]])

def load_data():
    print(f"Loading data from {DATA_FILE}")
    if not os.path.exists(DATA_FILE):
        return pd.DataFrame(columns=["X1", "X2", "y"])
    return pd.read_csv(DATA_FILE, nrows=10)

def save_data(df):
    df.to_csv(DATA_FILE, index=False)

def main():
    print("Gaussian Process Optimization Explorer")
    print("--------------------------------------")

    # 1. Load Data
    df = load_data()
    print(f"Loaded {len(df)} observations.")

    optimizer = GPOptimizer()

    while True:
        if not df.empty:
            X = df[["X1", "X2"]].values
            y = df["y"].values
            
            # 2. Fit Model
            print("Fitting GP model...")
            optimizer.fit(X, y)
            
            # 3. Optimize / Suggest Next Point
            print("Optimizing acquisition function (Expected Improvement)...")
            next_point = optimizer.suggest_next_point(BOUNDS)
            print(f"\n>>> SUGGESTED NEXT POINT: {next_point[0]:.6f}-{next_point[1]:.6f}")
            
            # 4. Visualize
            print("Generating plots...")
            plot_gp_search_space(optimizer, BOUNDS)
        else:
            print("No data available. Suggesting random starting point.")
            next_point = np.random.uniform(BOUNDS[:, 0], BOUNDS[:, 1])
            print(f"\n>>> SUGGESTED STARTING POINT: {next_point[0]:.6f}-{next_point[1]:.6f}")

        # 5. Interactive Input (Optional, for easy data entry)
        response = input("\nNew observation? [y/N]: ").strip().lower()
        if response == 'y':
            input_str = input("Enter input (format float-float, e.g. 0.5-0.5): ").strip()
            try:
                x_vals = GPOptimizer.parse_input(input_str)
                output_val = float(input("Enter output value: ").strip())
                
                new_row = {
                    "X1": x_vals[0],
                    "X2": x_vals[1],
                    "y": output_val
                }
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                df = df.drop_duplicates()
                if input("Save data? [y/N]: ").strip().lower() != 'y':
                    print("Data not saved.")
                    continue
                save_data(df)
                print("Data saved. Rerun the script to update model and suggestion.")
            except ValueError as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUser interrupted the program. Goodbye!")
    except Exception as e:
        print(f"\n\nAn error occurred: {e}")
        raise

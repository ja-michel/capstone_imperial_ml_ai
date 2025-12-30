import os
import numpy as np
import pathlib as pth
from typing import ClassVar
import sys
import pandas as pd

class DataLoader:
    PROJECT_DIR: ClassVar[pth.Path] = pth.Path(os.path.dirname(__file__)).parent.parent.parent
    DATA_DIR: ClassVar[pth.Path] = PROJECT_DIR / "data"
    DATASET_NAME: ClassVar[str] = "initial_data"
    DATASET_PATH: ClassVar[pth.Path] = DATA_DIR / DATASET_NAME
    DATA: ClassVar[dict] = {}

    @classmethod
    def init(cls):
        """Initialize the data loader by creating necessary directories."""
        if DataLoader.DATA_DIR.as_posix() not in sys.path:
            sys.path.append(DataLoader.DATA_DIR.as_posix())
        for f in range(1,9):
            function_name = f"f{f}"
            data = DataLoader.load_all_function_data(function_name=function_name)
            DataLoader.DATA[function_name] = data
            
    @staticmethod
    def to_pandas(*, data: dict) -> pd.DataFrame:
        col_names = [f'X_{i}' for i in range(data["X"].shape[1])] + ["y"]
        df = pd.DataFrame(np.hstack((data["X"], data["y"].reshape(-1, 1))), columns=col_names)
        return df
    
    @staticmethod
    def load_data(*, function_name: str, **kwargs):
        inputs_path = DataLoader.DATASET_PATH / function_name / "initial_inputs.npy"
        outputs_path = DataLoader.DATASET_PATH / function_name / "initial_outputs.npy"
        X = np.load(inputs_path)
        y = np.load(outputs_path)
        return {"X": X, "y": y}
    
    @staticmethod
    def load_submission_data(*, function_name: str, **kwargs):
        """Load submission data for a given function."""

        data_path = DataLoader.DATA_DIR / "submissions" / "data.py"
        
        with open(data_path, 'r') as f:
            # Read the content of the file
            content = f.read()
            # Execute the content to load the dictionaries
            exec(content, globals())

        # Access the dictionary corresponding to the function_name
        submission_data = globals()[function_name]

        X = np.array(submission_data["inputs"])
        y = np.array(submission_data["outputs"])
        return {"X": X, "y": y}
    
    @staticmethod
    def load_all_function_data(*, function_name: str, **kwargs):
        data = DataLoader.load_data(function_name=function_name)
        submission_data = DataLoader.load_submission_data(function_name=function_name)
        X = np.vstack((data["X"], submission_data["X"]))
        y = np.hstack((data["y"], submission_data["y"]))
        return {"X": X, "y": y}

DataLoader.init()
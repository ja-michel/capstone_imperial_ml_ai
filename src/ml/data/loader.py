import os
import numpy as np
import pathlib as pth
from typing import ClassVar

class DataLoader:
    PROJECT_DIR: ClassVar[pth.Path] = pth.Path(os.path.dirname(__file__)).parent.parent.parent
    DATA_DIR: ClassVar[pth.Path] = PROJECT_DIR / "data"
    DATASET_NAME: ClassVar[str] = "initial_data"
    DATASET_PATH: ClassVar[pth.Path] = DATA_DIR / DATASET_NAME

    @staticmethod
    def load_data(*, function_name: str, **kwargs):
        inputs_path = DataLoader.DATASET_PATH / function_name / "initial_inputs.npy"
        outputs_path = DataLoader.DATASET_PATH / function_name / "initial_outputs.npy"
        X = np.load(inputs_path)
        y = np.load(outputs_path)
        print(f"Loaded data from {DataLoader.DATASET_PATH / function_name}")
        print(f"Input shape: {X.shape}, Output shape: {y.shape}")
        return X, y

    @staticmethod
    def load_submission_data(*, function_name: str, **kwargs):
        submission_path = DataLoader.DATA_DIR / "submissions" / f"{function_name}.json"
        import json
        with open(submission_path, 'r') as f:
            submission_data = json.load(f)
        X = np.array(submission_data["inputs"])
        y = np.array(submission_data["outputs"])
        print(f"Loaded submission data from {submission_path}")
        print(f"Submission input shape: {X.shape}, output shape: {y.shape}")
        return X, y
    
    @staticmethod
    def load_all_function_data(*, function_name: str, **kwargs):
        X, y = DataLoader.load_data(function_name=function_name)
        X_submission, y_submission = DataLoader.load_submission_data(function_name=function_name)
        return np.vstack((X, X_submission)), np.hstack((y, y_submission))
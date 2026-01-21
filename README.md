# Professional Certificate in Machine Learning and Artificial Intelligence
### Project Overview
The goal of this project is to explore a range of techniques to develop models that solve the black-box optimization challenge. The black-box optimization problem is to maximize the output of an unknown function. A function can be unknown for many reasons, including an exceedingly high cost of evaluation, resource constraints, or complexity. In this exercise, one can only provide a limited number of inputs and observe the outputs.

These types of problems are relevant in applications where functions are too complex to resolve. For example, in fluid dynamics, these techniques help optimize shapes to minimize drag; in financial engineering, where no formal formulas are known to describe certain behaviours, black-box models are used to approximate a solution. Understanding the techniques used to solve these problems will help me have a deeper understanding of work in adjacent areas and collaborate more effectively with my colleagues.

### Inputs and Outputs
The project features eight different functions. Each function has an increasing number of input parameters. Each parameter is a rational number in the range of [0, 0.999999]. The output is also a rational number in the range [0, 0.999999].

The input parameters are entered as a string composed of the input values joined by a hyphen (-). For example, for a function with two input parameters, the query looks like this: 0.123456-0.654321.

The output represents the evaluation of the function given the input. The inputs and outputs are then added to the dataset to further refine the model and make subsequent predictions.

### Challenge Objectives
Through this project, I am gaining experience working with data and developing modeling techniques that are relevant to real-world problems. The goal is to maximize the output of each function using 13 submissions. Each submission is evaluated once per week.

Ten inputs and outputs are provided at the start of the task and serve as the starting point to develop models that will guide every subsequent submission.

### Technical Approach
In week one, I focused on developing an understanding of the data. I performed exploratory data analysis on the data given at the start of the task. My objective was to identify patterns, correlations, and outliers that could help me decide on the next steps.

In week two, I developed models for the first three functions that had a low dimensionality (2 and 3 parameters). I focused on Bayesian Optimization. I researched this technique, as it is recognized as one of the most effective for these types of problems. I experimented with different kernels and developed plots to understand the behavior of the model.

In week three, I experimented with other techniques, such as gradient descent. This helped me frame the problem from a different perspective and tackle the other functions where data patterns are less obvious and nonlinear relationships are more likely.


### Repository Structure
The repository is organized as follows:

```
├── data/
│   └── observations.csv    <- Stores the input-output pairs from the function evaluations.
├── docs/                   <- Documentation and reference materials.
├── models/                 <- Serialized models (if any).
├── notebooks/              <- Jupyter notebooks for exploration.
├── src/
│   ├── ml/                 <- Main package for the project.
│   │   ├── optimization.py <- Gaussian Process optimization logic.
│   │   └── visualization.py<- Plotting functions (Mean, Uncertainty, EI).
│   └── explore_script.py   <- Main script to run the exploration loop.
├── tests/                  <- Automated tests for the code.
└── README.md               <- The top-level README for developers using this project.
```

### Libraries and Technologies
-   **scikit-learn**: Used for `GaussianProcessRegressor`. Chosen for its ease of use and sufficient performance for the scale of this problem (small N).
-   **scipy**: Used for `minimize` (L-BFGS-B) to optimize the acquisition function.
-   **pandas**: Used for managing the observations data (CSV I/O).
-   **matplotlib**: Used for visualizing the GP search space and acquisition function.

### How to Run the Optimization
1.  **Environment Setup**: Ensure your virtual environment is active and dependencies are installed.
2.  **Run the Explorer**:
    ```bash
    export PYTHONPATH=$PYTHONPATH:$(pwd)/src
    .venv/bin/python src/explore_script.py
    ```
3.  **Workflow**:
    -   The script will load existing data from `data/observations.csv`.
    -   It fits a Gaussian Process model to the data.
    -   It displays a heatmap of the current model prediction, uncertainty, and Expected Improvement.
    -   It suggests the next point to query (format: `0.123-0.456`).
    -   You can then enter the result of the function evaluation to update the dataset.

## Professional Certificate in ML + AI
### Capstone Project

The project is a real-world style challenge in ML: optimising unknown functions using limited information. The objective is to use the different techniques seen in the course to develop models of the black-box functions and minimise their output.

[Project Description](./capstone.md)

#### Function Descriptions
- [Function 1](function_description.md#f1)
- [Function 2](function_description.md#f2)
- [Function 3](function_description.md#f3)
- [Function 4](function_description.md#f4)
- [Function 5](function_description.md#f5)
- [Function 6](function_description.md#f6)
- [Function 7](function_description.md#f7)
- [Function 8](function_description.md#f8)

---
#### Inputs + Outputs
Each parameters of the input for any function is a rational number in the range [0.000000, 0.999999] and are submitted as a string in the format `[input1]-[input2]-...-[inputN]`. For example, the input for a 2d function can be `0.123456-0.654321`.

Each function is evaluated once every week with the submitted inputs. The result of the evaluation is provided when the functions are evaluated.

The inputs and outputs from each submission can be incorporated into the dataset to increase the number of samples and refine the models before submitting a new set of inputs.

### Progress Log
Week 1
Focused on lower dimension functions f1 and f2. Used these functions to practice exploratory data analysis and Bayesian Optimisation to improve my understanding of the technique and build a process that I can later improve and apply to other functions. 
Experimented with different kernels: ...  and added noise. 

Findings:
- outliers
- noisy data
- gp plot + ei func plot
- plot to visualise the kernel
- Matern(nu=2.5): The 5/2 kernel we discussed.
- WhiteKernel: This handles "Noise". It tells the model "The data isn't perfect, allow for some wiggle room."

Notebook:
[notebook f1](../notebooks/f1.ipynb)
[notebook f2](../notebooks/f2.ipynb)

Week 2
Implemented Bayesian Optimisation for all other functions. 

### Change Log

#### Imperial College Business School


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class PlotUtils:
   
   
    @staticmethod
    def scatter(X: np.ndarray, y: np.ndarray) -> None:
        plt.figure(figsize=(8, 6))
        for i in range(X.shape[1]):
            sns.scatterplot(x=X[:, i], y=y, label=f'X[:, {i}]', alpha=0.6, s=50)
        plt.xlabel('X values')
        plt.ylabel('y')
        plt.title('Scatter plots of X columns vs y')
        plt.legend()
        plt.show()
    
    @staticmethod
    def box(data: np.ndarray,) -> None:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data)
        if data.ndim == 1:
            plt.xticks(ticks=[0], labels=['y'])
        else:
            plt.xticks(ticks=range(data.shape[1]), labels=[f'X[:, {i}]' for i in range(data.shape[1])])
        # plt.xticks(ticks=range(X.shape[1] + 1), labels=[f'X[:, {i}]' for i in range(X.shape[1])] + ['y'])
        plt.title('Box plots')
        plt.show()
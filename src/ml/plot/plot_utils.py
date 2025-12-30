import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class PlotUtils:
   
   
    @staticmethod
    def scatter(X: np.ndarray, y: np.ndarray, figsize:tuple[int, int] = (8, 6), title: str = 'Scatter plots of X columns vs y', show_min_max: bool = True) -> None:
        plt.figure(figsize=figsize)
        for i in range(X.shape[1]) if X.ndim > 1 else range(1):
            x = X if X.ndim == 1 else X[:, i]
            sns.scatterplot(x=x, y=y, label=f'X[:, {i}]', alpha=0.6, s=50)
            
            if show_min_max:
                min_y_idx = np.argmin(y)
                max_y_idx = np.argmax(y)
                plt.scatter(x[min_y_idx], y[min_y_idx], color='red', alpha=0.6, s=50, marker='o', label=f'Min y: {y[min_y_idx]:.6f} @ x={x[min_y_idx]:.6f}')
                plt.scatter(x[max_y_idx], y[max_y_idx], color='green', alpha=0.6, s=50, marker='o', label=f'Max y: {y[max_y_idx]:.6f} @ x={x[max_y_idx]:.6f}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(title)
        plt.legend()
        plt.show()
    
    @staticmethod
    def box(data: np.ndarray, figsize:tuple[int, int] = (8, 6), title: str = 'Box plots') -> None:
        plt.figure(figsize=figsize)
        sns.boxplot(data=data)
        if data.ndim == 1:
            plt.xticks(ticks=[0], labels=['y'])
        else:
            plt.xticks(ticks=range(data.shape[1]), labels=[f'X[:, {i}]' for i in range(data.shape[1])])
        # plt.xticks(ticks=range(X.shape[1] + 1), labels=[f'X[:, {i}]' for i in range(X.shape[1])] + ['y'])
        plt.title(title)
        plt.show()

    @staticmethod
    def histogram(data: np.ndarray, bins: int = 30) -> None:
        plt.figure(figsize=(8, 6))
        if data.ndim == 1:
            sns.histplot(data, bins=bins, kde=True)
            plt.xlabel('y values')
        else:
            for i in range(data.shape[1]):
                sns.histplot(data[:, i], bins=bins, kde=True, label=f'X[:, {i}]', alpha=0.6)
            plt.xlabel('X values')
            plt.legend()
        plt.title('Histograms')
        plt.show()
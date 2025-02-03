import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split

def get_kfold_data(data:pd.DataFrame, k:int, reproducibility_seed:int=42):
    kfold_split = KFold(n_splits=k, shuffle=True, random_state=reproducibility_seed)

    kfold_data = []
    for i, (train_and_val_index, test_index) in enumerate(kfold_split.split(data)):
        training_and_val = data.iloc[train_and_val_index]
        train, val = train_test_split(training_and_val, test_size=0.2, random_state=reproducibility_seed)
        test = data.iloc[test_index]
        fold_data = {"train": train, "val": val, "test": test}

        print(f"Fold: {i}/{k}")
        print(f"Train shape: {train.shape} | {train.shape[0] / data.shape[0] * 100:.2f}%")
        print(f"Validation shape: {val.shape} | {val.shape[0] / data.shape[0] * 100:.2f}%")
        print(f"Test shape: {test.shape} | {test.shape[0] / data.shape[0] * 100:.2f}%")
        print()

        kfold_data.append(fold_data)
    
    return kfold_data

def print_statistics(data:pd.DataFrame, column:str) -> None:
    """
    Calculates and prints the mean, median, standard deviation,
    minimum and maximum values of a column in a DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing the data.
        column (str): The name of the column to calculate the statistics for.
    """

    mean = data.mean()
    median = data.median()
    std = data.std()
    min_val = data.min()
    max_val = data.max()

    print(f"Statistics for column: {column}")
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Standard Deviation: {std}")
    print(f"Minimum Value: {min_val}")
    print(f"Maximum Value: {max_val}")
    print()

def plot_distribution(data:pd.DataFrame, column:str, title:str) -> None:
    """
    Plots the distribution of values in a column of a DataFrame.

    Args:
        data (pd.DataFrame): The DataFrame containing the data to plot.
        column (str): The column to plot.
        title (str): The title of the plot.
    """
    data.plot.hist(title=title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()
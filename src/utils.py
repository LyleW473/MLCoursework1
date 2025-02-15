import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold, train_test_split
from typing import List, Dict

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

def calculate_metrics(targets:np.ndarray, preds:np.ndarray) -> Dict[str, float]:
    """
    Calculates the Mean Absolute Error, Mean Squared Error, Root Mean Squared Error,
    Pearson Correlation Coefficient and Spearman Rank Correlation Coefficient of a model.

    Args:
        targets (np.ndarray): The true values.
        preds (np.ndarray): The predicted values.
    """
    mae = mean_absolute_error(y_true=targets, y_pred=preds)
    mse = mean_squared_error(y_true=targets, y_pred=preds)
    rmse = np.sqrt(mse).item()
    
    temp_outputs = np.array(preds).flatten()
    temp_targets = np.array(targets).flatten()
    pcc, _ = pearsonr(temp_outputs, temp_targets)
    spearman_r = spearmanr(temp_outputs, temp_targets)[0]
    r2_score = calculate_r2_score(y_pred=preds, y_true=targets)
    
    metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "pcc": pcc,
            "spearman_r": spearman_r,
            "r2_score": r2_score
            }
    return metrics

def calculate_r2_score(y_pred, y_true) -> float:
    """
    Calculates the R^2 score of a model.

    Args:
        y_pred (np.ndarray): The predicted values.
        y_true (np.ndarray): The true values.
    """
    eps = y_true - y_pred
    rss = np.sum(eps ** 2)
    tss = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2

def convert_non_numeric_to_numeric(data:pd.DataFrame) -> pd.DataFrame:
    """
    Converts non-numeric columns in a DataFrame to numeric
    using a mix of one-hot encoding and ordinal encoding.

    Args:
        data (pd.DataFrame): The DataFrame containing the data to convert.
    """
    colours = data["color"].value_counts().index.tolist()
    print(colours)

    for colour in colours:
        data[f"colour_{colour}"] = data["color"].apply(lambda x: 1 if x == colour else 0)
    data.drop("color", axis=1, inplace=True)

    quality_mappings = {"Ideal": 0, "Premium": 1, "Very Good": 2, "Good": 3, "Fair": 4}
    data["cut"] = data["cut"].map(quality_mappings)

    clarity_mappings = {"IF": 0, "VVS1": 1, "VVS2": 2, "VS1": 3, "VS2": 4, "SI1": 5, "SI2": 6, "I1": 7}
    data["clarity"] = data["clarity"].map(clarity_mappings)
    print(data)
    return data


def get_kfold_data(data:pd.DataFrame, k:int, reproducibility_seed:int=42) -> List[Dict[str, pd.DataFrame]]:
    """
    Splits the data into k-folds for cross-validation.

    Args:
        data (pd.DataFrame): The data to split.
        k (int): The number of folds to split the data into.
        reproducibility_seed (int): The seed to use for reproducibility.
    """
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
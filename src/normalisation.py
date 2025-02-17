import pandas as pd
from typing import Union

class Normaliser:
    def calculate_mean(self, data:pd.DataFrame) -> float:
        """
        Calculates the mean of a DataFrame.

        Args:
            data (pd.DataFrame): The data to calculate the mean of.
        """
        mean = data.mean()
        return mean

    def calculate_std(self, data:pd.DataFrame) -> float:
        """
        Calculates the standard deviation of a DataFrame.

        Args:
            data (pd.DataFrame): The data to calculate the standard deviation of.
        """
        std = data.std()
        return std
    
    def standardise(self, data:pd.DataFrame, mean:Union[pd.Series, None]=None, std:Union[pd.Series, None]=None) -> pd.DataFrame:
        """
        Standardises a DataFrame by subtracting the mean and dividing by the standard deviation.

        Args:
            data (pd.DataFrame): The data to standardise.
        """
        if mean is None:
            mean = data.mean()
        if std is None:
            std = data.std()
        data = (data - mean) / (std + 1e-8)
        return data
import pandas as pd

class Normaliser:
    def standardise(self, data:pd.DataFrame) -> pd.DataFrame:
        """
        Standardises a DataFrame by subtracting the mean and dividing by the standard deviation.

        Args:
            data (pd.DataFrame): The data to standardise.
        """
        mean = data.mean()
        std = data.std()
        data = (data - mean) / (std + 1e-8)
        return data
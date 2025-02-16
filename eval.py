import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import os

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split

from src.utils import get_kfold_data, convert_non_numeric_to_numeric, calculate_r2_score, calculate_metrics
from src.normalisation import Normaliser
from src.constants import *


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)

    # Find columns
    all_columns = data.columns.tolist()
    print(all_columns)

    numeric_columns = data.select_dtypes(include=["number"]).columns.tolist()
    numeric_columns.remove("outcome") # Remove the target column
    print(numeric_columns)

    non_numeric_columns = data.select_dtypes(exclude=["number"]).columns.tolist()
    print(non_numeric_columns)
    
    # Converting non-numeric features to numerical features
    data = convert_non_numeric_to_numeric(data=data)
    print(data)

    # Standardising the data
    normaliser = Normaliser()
    for column in numeric_columns:
        print(data[column])
        data[column] = normaliser.standardise(data[column])
        print("after", data[column])

    best_hyperparameters = {
        "objective":"reg:squarederror", 
        "n_estimators": 1000,
        "max_depth": 5,
        "learning_rate": 0.1, 
        "random_state": REPRODUCIBILITY_SEED
    } # TODO: Load the best hyperparameters from the hyperparameter tuning process
    
    model = xgb.XGBRegressor(**best_hyperparameters)

    # ----------------------------------------------------------------
    # Base code:

    # Random 80/20 train/validation split
    trn, tst = train_test_split(data, test_size=0.2, random_state=123)

    # Train a linear model
    X_trn = trn.drop(columns=['outcome'])
    y_trn = trn['outcome']
    X_tst = tst.drop(columns=['outcome'])
    y_tst = tst['outcome']

    model.fit(X_trn, y_trn)
    yhat_lm = model.predict(X_tst)

    # Format submission
    out = pd.DataFrame({'yhat': yhat_lm})
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    save_path = f'{SUBMISSIONS_DIR}/CW1_submission.csv'
    out.to_csv(save_path, index=False)

    # Read in the submission
    yhat_lm = np.array(pd.read_csv(save_path)['yhat'])

    # This is the R^2 function
    def r2_fn(yhat):
        eps = y_tst - yhat
        rss = np.sum(eps ** 2)
        tss = np.sum((y_tst - y_tst.mean()) ** 2)
        r2 = 1 - (rss / tss)
        return r2

    # Evaluate
    print(r2_fn(yhat_lm))
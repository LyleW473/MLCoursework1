import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import numpy as np
import os
import json

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split

from src.utils import get_kfold_data, convert_non_numeric_to_numeric, calculate_r2_score, calculate_metrics
from src.normalisation import Normaliser
from src.constants import *


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    test_data = pd.read_csv(TEST_DATA_PATH)

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
    test_data = convert_non_numeric_to_numeric(data=test_data)
    print(test_data)

    # Split data into training and test sets
    train_data, local_test_data = train_test_split(data, test_size=0.2, random_state=REPRODUCIBILITY_SEED)
    print(f"Training set size: {len(train_data)} | Test set size: {len(local_test_data)}")    
    print()

    # Standardising the data
    with open(f"{TRAINING_STATISTICS_DIR}/stats.json", "r") as f:
        stats_for_each_column = json.load(f)
    print(stats_for_each_column)

    normaliser = Normaliser()
    for column in numeric_columns:
        print(data[column])
        train_data_column_stats = stats_for_each_column[column]
        train_data_column_mean = train_data_column_stats["mean"]
        train_data_column_std = train_data_column_stats["std"]

        train_data[column] = normaliser.standardise(train_data[column], mean=train_data_column_mean, std=train_data_column_std)
        stats_for_each_column[column] = {
            "mean": train_data_column_mean,
            "std": train_data_column_std
        }

        # Normalise (local) test set using the mean and std of the training data
        local_test_data[column] = normaliser.standardise(local_test_data[column], mean=train_data_column_mean, std=train_data_column_std)
        print("after", train_data[column])

        # Normalise the test data using the mean and std of the training data
        test_data[column] = normaliser.standardise(test_data[column], mean=train_data_column_mean, std=train_data_column_std)

    # Get the K-Fold data for training best hyperparameter models.
    kfold_data = get_kfold_data(data=train_data, k=NUM_FOLDS, reproducibility_seed=REPRODUCIBILITY_SEED)
    
    # Create a model for each fold.
    local_test_predictions_per_model = {}
    test_predictions_per_model = {}
    local_test_data_y = np.array(local_test_data["outcome"])
    local_test_data_x = local_test_data.drop(columns=["outcome"])
    print(local_test_data_y)

    for fold in range(NUM_FOLDS):
        with open(f"{BEST_HYPERPARAMETERS_DIR}/xgb/fold_{fold+1}.json", "r") as f:
            best_hyperparameters = json.load(f)

        with open(f"{BEST_HYPERPARAMETERS_DIR}/xgb/fold_{fold+1}_selected_features.json", "r") as f:
            best_selected_features = json.load(f) 

        # Extract data for the fold
        fold_data = kfold_data[fold]
        fold_train_data = fold_data["train"]
        fold_val_data = fold_data["val"]

        fold_train_y = fold_train_data["outcome"]
        fold_val_y = np.array(fold_val_data["outcome"])
        
        fold_train_x = fold_train_data.drop(columns=["outcome"])
        fold_val_x = fold_val_data.drop(columns=["outcome"])

        fold_train_x = fold_train_x[best_selected_features]
        fold_val_x = fold_val_x[best_selected_features]

        local_test_data_x_for_model = local_test_data_x[best_selected_features]
        test_data_x_for_model = test_data[best_selected_features]

        # Train the model
        fold_model = xgb.XGBRegressor(**best_hyperparameters)
        fold_model.fit(fold_train_x, fold_train_y)
        val_preds = fold_model.predict(fold_val_x)

        # Calculate metrics
        metrics = calculate_metrics(targets=fold_val_y, preds=val_preds)
        mae = metrics["mae"]
        mse = metrics["mse"]
        rmse = metrics["rmse"]
        pcc = metrics["pcc"]
        spearman_r = metrics["spearman_r"]
        r2_score = metrics["r2_score"]

        print(f"Fold: {fold+1}/{NUM_FOLDS}")
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"PCC: {pcc}")
        print(f"Spearman R: {spearman_r}")
        print(f"R2 Score: {r2_score}")
        print()

        # Predict on the local test set
        local_test_preds = fold_model.predict(local_test_data_x_for_model)
        local_test_predictions_per_model[f"fold_{fold+1}"] = local_test_preds

        local_test_metrics = calculate_metrics(targets=local_test_data_y, preds=local_test_preds)
        local_test_mae = local_test_metrics["mae"]
        local_test_mse = local_test_metrics["mse"]
        local_test_rmse = local_test_metrics["rmse"]
        local_test_pcc = local_test_metrics["pcc"]
        local_test_spearman_r = local_test_metrics["spearman_r"]
        local_test_r2_score = local_test_metrics["r2_score"]
        
        print(f"Local Test Set Metrics")
        print(f"MAE: {local_test_mae}")
        print(f"MSE: {local_test_mse}")
        print(f"RMSE: {local_test_rmse}")
        print(f"PCC: {local_test_pcc}")
        print(f"Spearman R: {local_test_spearman_r}")
        print(f"R2 Score: {local_test_r2_score}")
        print()

        # Predict on hidden test set
        test_preds = fold_model.predict(test_data_x_for_model)
        test_predictions_per_model[f"fold_{fold+1}"] = test_preds

    # ----------------------------------------------------------------
    # Base code:

    # Checking the metrics for bagging
    local_test_predictions = pd.DataFrame(local_test_predictions_per_model)
    print(local_test_predictions)
    local_test_predictions["yhat"] = local_test_predictions.mean(axis=1)
    local_test_predictions.drop(columns=[f"fold_{i+1}" for i in range(NUM_FOLDS)], inplace=True)
    print(local_test_data_y)
    local_test_predictions["actual"] = local_test_data_y
    print(local_test_predictions)

    metrics = calculate_metrics(targets=local_test_predictions["actual"], preds=local_test_predictions["yhat"])
    mae = metrics["mae"]
    mse = metrics["mse"]
    rmse = metrics["rmse"]
    pcc = metrics["pcc"]
    spearman_r = metrics["spearman_r"]
    r2_score = metrics["r2_score"]

    print(f"Bagging Metrics")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"PCC: {pcc}")
    print(f"Spearman R: {spearman_r}")
    print(f"R2 Score: {r2_score}")
    print()

    # Generate final predictions by averaging the predictions from each model
    out = pd.DataFrame(test_predictions_per_model)
    print(out)
    out["yhat"] = out.mean(axis=1)
    out.drop(columns=[f"fold_{i+1}" for i in range(NUM_FOLDS)], inplace=True)
    print(out)

    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    save_path = f'{SUBMISSIONS_DIR}/CW1_submission_K22039642.csv'
    out.to_csv(save_path, index=False)

    # # Read in the submission with the actual outcomes
    # actual_tst = pd.read_csv(f"INSERT-PATH-TO-TEST-SET-WITH-ACTUAL-OUTCOMES")
    # yhat_lm = np.array(pd.read_csv(save_path)['yhat'])

    # # This is the R^2 function
    # def r2_fn(yhat):
    #     eps = actual_tst - yhat
    #     rss = np.sum(eps ** 2)
    #     tss = np.sum((actual_tst - actual_tst.mean()) ** 2)
    #     r2 = 1 - (rss / tss)
    #     return r2

    # # Evaluate
    # print(r2_fn(yhat_lm))
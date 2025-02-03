import pandas as pd
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
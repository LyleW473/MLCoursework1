# Paths
MAIN_DATA_DIR = "data"
MAIN_DATA_FILE = "CW1_train.csv"
TEST_DATA_FILE = "CW1_test.csv"
DATA_PATH = f"{MAIN_DATA_DIR}/{MAIN_DATA_FILE}"
TEST_DATA_PATH = f"{MAIN_DATA_DIR}/{TEST_DATA_FILE}"
SUBMISSIONS_DIR = f"submissions"
BEST_HYPERPARAMETERS_DIR = f"model_best_hyperparameters"
TRAINING_STATISTICS_DIR = f"training_statistics"

# Constants
EPS = 1e-8
REPRODUCIBILITY_SEED = 42
NUM_FOLDS = 5
N_TRIALS = 100 # Number of trials for hyperparameter tuning
USE_FEATURE_SELECTION = True
REMOVE_OUTLIERS = True
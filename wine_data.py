"""
Load and split the red wine quality dataset.

Both the training sweep (linear_regression.ipynb) and the endpoint check (test_endpoint.py)
use this module, so they always see exactly the same test set.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

CSV_URL = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
TARGET = "quality"
RANDOM_STATE = 40


def load_wine_data(url=CSV_URL):
    return pd.read_csv(url, sep=";")


def split_wine_data(data):
    """Return train_x, test_x, train_y, test_y with a fixed 75/25 split."""
    train, test = train_test_split(data, test_size=0.25, random_state=RANDOM_STATE)
    train_x = train.drop(columns=[TARGET])
    test_x = test.drop(columns=[TARGET])
    return train_x, test_x, train[[TARGET]], test[[TARGET]]

"""
Check that the model served at the REST endpoint is the registered champion.

Sends the wine test set to the running `mlflow models serve` endpoint, loads the champion
model locally from the Model Registry, and checks that both give the same predictions.
Exits with code 0 on success and 1 on a mismatch.

Usage (from the mlflow_lab1 folder, with the model server already running):
    python test_endpoint.py [--url http://127.0.0.1:5001/invocations]
"""

import argparse
import sys

import numpy as np
import requests
from sklearn.metrics import mean_squared_error

import mlflow
from mlflow import MlflowClient
from wine_data import load_wine_data, split_wine_data

MODEL_NAME = "ElasticnetWineModel"
ALIAS = "champion"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:5001/invocations")
    args = parser.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    champion = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
    print(f"Champion: {MODEL_NAME} version {champion.version} (run {champion.run_id})")

    _, test_x, _, test_y = split_wine_data(load_wine_data())

    # Same request format the MLflow scoring server documents for pandas DataFrames
    response = requests.post(args.url, json={"dataframe_split": test_x.to_dict(orient="split")}, timeout=30)
    response.raise_for_status()
    served = np.asarray(response.json()["predictions"]).ravel()

    local = np.asarray(mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}").predict(test_x)).ravel()

    rmse = np.sqrt(mean_squared_error(test_y, served))
    print(f"Rows scored by endpoint: {len(served)}")
    print(f"First 5 served predictions: {np.round(served[:5], 3).tolist()}")
    print(f"RMSE of served predictions on test set: {rmse:.4f}")

    if served.shape == local.shape and np.allclose(served, local):
        print("PASS: served predictions match the champion model loaded from the registry")
        return 0
    print("FAIL: served predictions differ from the champion model loaded from the registry")
    return 1


if __name__ == "__main__":
    sys.exit(main())

import warnings

import os
import xgboost as xgb
import requests

from urllib.parse import urlparse
from sklearn.datasets import load_svmlight_file

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    train_dataset_path = "mlflow-xgboost/src/agaricus.txt.train"
    test_dataset_path = "mlflow-xgboost/src/agaricus.txt.test"

    # NOTE: Workaround to load SVMLight files from the XGBoost example
    X_train, y_train = load_svmlight_file(train_dataset_path)
    X_test, y_test = load_svmlight_file(test_dataset_path)
    X_train = X_train.toarray()
    X_test = X_test.toarray()

    print("**************************")
    print("X_train")
    print(X_train)

    print("**************************")
    print("y_train")
    print(y_train)

    print("**************************")
    # read in data
    dtrain = xgb.DMatrix(data=X_train, label=y_train)

    print("**************************")
    print("dtrain")
    print(dtrain)


    print("**************************")
    x_0 = X_test[0:1]
    print("x_0")
    print(x_0)

    print("**************************")
    print("x_0.shape", "x_0.tolist()")
    print(x_0.shape)
    print(x_0.tolist())
    
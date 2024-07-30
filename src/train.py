# %load src/train.py
# Original source code and more details can be found in:
# https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html

# The data set used in this example is from
# http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties.
# In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# def accuracy(actual, pred):
#     acc = roc_auc_score(actual, pred, multi_class="ovo")
#     return acc 

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # # Read the wine-quality csv file from the URL
    # csv_url = (
    #     "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    # )
    # try:
    #     data = pd.read_csv(csv_url, sep=";")
    # except Exception as e:
    #     logger.exception(
    #         "Unable to download training & test CSV, "
    #         "check your internet connection. Error: %s",
    #         e,
    #     )

    data = pd.read_csv('winequality-red.csv', sep=";")
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha =  0.4
    l1_ratio =  0.2
    
    # mlflow.set_tracking_uri(uri="http://mlflow-nginx.apps.cluster-tdb7n.tdb7n.sandbox2943.opentlc.com")
    mlflow.set_tracking_uri(uri="http://mlflow-auth-tracon-xxiv-mbahmani-0.apps.ocp.solutioncenter-munich.de")
    mlflow.set_experiment("wine quality - trainin and validation/testing accuracy")

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # acc_training = accuracy(train_y, lr.predict(train_x))
        # acc_testing = accuracy(test_y, lr.predict(test_x))
        # print("  acc_training: %s" % acc_training)
        # print("  acc_testing: %s" % acc_testing)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
        # mlflow.log_metric("acc_training", acc_training)
        # mlflow.log_metric("acc_testing", acc_testing)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        model_signature = infer_signature(train_x, train_y)

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry,
            # which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(
                lr,
                "model",
                registered_model_name="ElasticnetWineModel",
                signature=model_signature,
            )
        else:
            mlflow.sklearn.log_model(lr, "model", signature=model_signature)

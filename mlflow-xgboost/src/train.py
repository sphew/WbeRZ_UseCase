import warnings
import os

import mlflow
import mlflow.xgboost
from mlflow.models.signature import infer_signature
from mlflow.server.auth.client import AuthServiceClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from xgboost import XGBClassifier

import logging

import json

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Start a MLFlow Experiment
# tracking_uri = "https://kilabor-playground.it.nrw.de:80/mlflow"
# # client = AuthServiceClient()
# client = mlflow.MlflowClient(tracking_uri=tracking_uri)
# print(client)
# for model in client.search_registered_models(filter_string="name LIKE 'tracking%'"):
#     for model_version in model.latest_versions:
#         print(f"name={model_version.name}; run_id={model_version.run_id}; version={model_version.version}, stage={model_version.current_stage}")
# mlflow.set_experiment(experiment_name="heart-condition-classifier")

# Test connection to remote MLFlow
current_tracking_url = mlflow.get_tracking_uri()
print(current_tracking_url)
# client = AuthServiceClient()
client = mlflow.MlflowClient()
# print(client)
# print(os.getenv("MLFLOW_TRACKING_USERNAME"))
# print(os.getenv("MLFLOW_TRACKING_PASSWORD"))

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # Download and load the data-set
    # file_url = "https://azuremlexampledata.blob.core.windows.net/data/heart-disease-uci/data/heart.csv"
    file_url = "mlflow-xgboost/src/heart.csv"
    df = pd.read_csv(file_url)

    print(df.head(5))

    # Use encoded values for categorical variables
    df["thal"] = df["thal"].astype("category").cat.codes

    print(df["thal"].unique())

    # Split the data into training and test sets. (0.7, 0.3) split.
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("target", axis=1), df["target"], test_size=0.3
    )

    print("X_test - y_test")
    print(X_test.head(2))
    print(y_train.head(2))

    # Start a MLFlow Experiment
    mlflow.set_experiment(experiment_name="heart-condition-classifier new")
    
    # Start tracking training-run with mlflow
    with mlflow.start_run() as run:
        # Logging parameters and metrices
        mlflow.xgboost.autolog()

        # Create a simple classifier
        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

        # Model Training
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Prediction
        y_pred = model.predict(X_test)

        # Calculate relevant metrices
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        print("Recall: %.2f%%" % (recall * 100.0))
        
        model_signature = infer_signature(X_train, y_train)

        # Log model
        mlflow.xgboost.log_model(
            xgb_model=model, 
            artifact_path="model", 
            registered_model_name="heartClassifier",
            signature=model_signature,
            input_example=X_test[0:1]
        )

        mlflow.log_metric("Accuracy", (accuracy * 100.0))
        mlflow.log_metric("Recall", (recall * 100.0))

# Explore the logged info
print(run.info)

run = mlflow.get_run(run.info.run_id)

print(pd.DataFrame(data=[run.data.params], index=["Value"]).T)

print(pd.DataFrame(data=[run.data.metrics], index=["Value"]).T)


import argparse
import os
import json

from typing import Dict
from kserve import Model, ModelServer, model_server, InferRequest, InferOutput, InferResponse, logging
from kserve.util.utils import generate_uuid

import pandas as pd 
import numpy as np 

import mlflow

ARTIFACT_SAVE_DIR = "./model_dir"

class MyModel(Model):
    def __init__(self, name: str):
       super().__init__(name)
       self.name = name
       self.model = None
    #    self.scaler = None
       self.ready = None
       self.load()

    def load(self):
        self.model = mlflow.sklearn.load_model(model_uri="files://model_dir/")
        self.ready = True
        
    def predict(self, payload: Dict, headers: Dict[str, str] = None):
        input_data = payload["inputs"]["data"]
        output = self.model(input_data)
        return {"predictions": output}

parser = argparse.ArgumentParser(parents=[model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    if args.configure_logging:
        logging.configure_logging(args.log_config_file)  # Configure kserve and uvicorn logger
    model = MyModel(args.model_name)
    kserve.ModelServer().start([model])
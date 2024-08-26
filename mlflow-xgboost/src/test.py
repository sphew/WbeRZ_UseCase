import mlflow
from mlflow import MlflowClient
import numpy as np
import pandas as pd

import os

# inputs = [age=54,sex=1,cp=4,trestbps=122,chol=286,fbs=0,restecg=2,thalach=116,exang=1,oldpeak=3.2,slope=2,ca=2,thal=3]
# inputs = np.array([54, 1, 4, 122, 286, 0, 2, 116, 1, 3.2, 2, 2, 3], dtype=np.int32)
# inputs = [54, 1, 4, 122, 286, 0, 2, 116, 1, 3.2, 2, 2, 3]
# inputs = np.array({'age': 54,'sex': 1,'cp': 4,'trestbps': 122,'chol': 286,'fbs': 0,'restecg': 2,'thalach': 116,'exang': 1,'oldpeak': 3.2,'slope': 2,'ca': 2,'thal': 3}, dtype=np.int32)
inputs= pd.DataFrame(data=[[54,1,4,122,286,0,2,116,1,3.2,2,2,3]], columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"])
# inputs['age'] = inputs['age'].astype('int32')

print("inputs.dtypes ", inputs.dtypes)

processed_inputs = inputs.copy(deep=True)
for cl in processed_inputs.columns:
    if cl == 'oldpeak':
        processed_inputs[cl] = processed_inputs[cl].astype('float32')
    else:
        processed_inputs[cl] = processed_inputs[cl].astype('int32')
print("processed_inputs.dtypes ", processed_inputs.dtypes)
print("inputs.dtypes ", inputs.dtypes)

print('************************************************************')
print(' PREDICTION on original Model --> heartClassifier')
print('************************************************************')
MODEL_URI = "models:/heartClassifier/1"
loaded_model = mlflow.pyfunc.load_model(MODEL_URI)

print('************************************************************')
print("model._model_meta._signature: ", loaded_model._model_meta._signature)
print('************************************************************')
print("model: ", loaded_model)
print('************************************************************')
print("--->>> PREDICTION on original Model: ", loaded_model.predict(processed_inputs))

print('************************************************************')

print('************************************************************')
print(' PREDICTION on Pre/PostProcessed Model --> heartClassifierPP')
print('************************************************************')
MODEL_URI_PP = "models:/heartClassifierPP/10"
loaded_modelPP = mlflow.pyfunc.load_model(MODEL_URI_PP)

print('************************************************************')
print("model._model_meta._signature: ", loaded_modelPP._model_meta._signature)
print('************************************************************')
print("model: ", loaded_modelPP)
print('************************************************************')
print("--->>> PREDICTION on Pre/PostProcessed Model: ", loaded_modelPP.predict(inputs))

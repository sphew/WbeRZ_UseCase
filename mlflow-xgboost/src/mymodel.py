
import mlflow
from mlflow.models import infer_signature
import pandas as pd
# from preprocessing_utils.preprocessText import preprocessText

# Define a custom PythonModel 
class heartClassifierPreProcessed(mlflow.pyfunc.PythonModel):
    
    def __init__(self):
        self.model = None
    
    def load_context(self, context):
        print("It is load_context function")
        """
        Load the model from the specified artifacts directory.
        """
        model_file_path = context.artifacts["model_file"]

        # Load the model
        # print("model_file_path:", model_file_path)
        self.model = mlflow.pyfunc.load_model(model_file_path)
        # print("self.model:", self.model)

    @staticmethod
    def _format_input(model_input):

        print("model_input.dtypes before: ", model_input.dtypes)
        print("model_input:", model_input)

        for cl in model_input.columns:
            if cl == 'oldpeak':
                model_input[cl] = model_input[cl].astype('float32')
            else:
                model_input[cl] = model_input[cl].astype('int32')

        print("model_input.dtypes after: ", model_input.dtypes)
        print("model_input ", model_input)

        return model_input

    @staticmethod
    def _format_output(prediction):

        if prediction == "1":
            return "You may have heart problems! Please let yourseld checked!"
        else:
            return "Your heart works like a rock!"

    def predict(self, context, model_input):
        print("It is predict function")
        
        """
        Perform prediction using the loaded model.
        """
        if self.model is None:
            raise ValueError(
                "The model has not been loaded. "
                "Ensure that 'load_context' is properly executed."
            )
        
        print("self.model:", self.model)
        # print("model_input:", model_input)
        
        # print("model_input ", model_input)
        # print(model_input.dtypes)

        # for cl in model_input.columns:
        #     if cl == 'oldpeak':
        #         model_input[cl] = model_input[cl].astype('float32')
        #     else:
        #         model_input[cl] = model_input[cl].astype('int32')

        # print(model_input.dtypes)
        # print("model_input ", model_input)

        prediction = self.model.predict(self._format_input(model_input))
        print("prediction ", prediction)
        # result = f"my custom model model prediction is: {prediction}"
        return self._format_output(prediction)

# Now we can store/log the preprocessed model to mlflow and use it for inferencing
# We start a new experiment
mlflow.set_experiment("heart-condition-classifier with Preprocessing new")

heartClassifierPreProcessed = heartClassifierPreProcessed()
inputs = pd.DataFrame(data=[[54,1,4,122,286,0,2,116,1,3.2,2,2,3]], columns=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"])
signature = infer_signature(
    model_input=inputs,
    model_output="it should be a string"
)
with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        python_model=heartClassifierPreProcessed,
        artifact_path="preprocessed_model",
        artifacts={"model_file": "models:/heartClassifier/1"},
        signature=signature,
        # input_example=X_train,
        # code_path=["mlflow-xgboost/src/preprocessing_utils"],
        registered_model_name="heartClassifierPP"  # <--- heartClassifierPreProcessed
    )
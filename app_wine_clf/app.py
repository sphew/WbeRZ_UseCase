from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Define the model server endpoint
ENDPOINT_URL = os.getenv("ENDPOINT_URL", "http://default-endpoint")
MLFLOW_MODEL_ENDPOINT = f"{ENDPOINT_URL}/invocations"

# Categories list
CATEGORIES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol"
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    message = None
    if request.method == "POST":
        # Get input data
        input_data = {category: float(request.form.get(category, 0)) for category in CATEGORIES}
        actual_class = request.form.get("actual_class")

        # Prepare data for inference
        df_test_input = pd.DataFrame([input_data])
        inference_request = {
            "dataframe_split": df_test_input.to_dict(orient="split")
        }

        try:
            # Send inference request to the model endpoint
            response = requests.post(MLFLOW_MODEL_ENDPOINT, json=inference_request)
            response.raise_for_status()
            prediction = response.json()["predictions"][0]

            # Check if the prediction matches the actual class
            if actual_class:
                if int(prediction) == int(actual_class):
                    message = "The model correctly predicted the wine quality!"
                else:
                    message = "The model did not deliver the correct class! Maybe next time!"
        except Exception as e:
            message = f"Error communicating with the model server: {str(e)}"

    return render_template("index.html", categories=CATEGORIES, prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

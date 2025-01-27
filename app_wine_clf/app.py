from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Define the model server endpoint
BASE_URL = "http://test-clf-app-test.apps.cluster-db46l.dynamic.redhatworkshops.io"
MLFLOW_MODEL_ENDPOINT = f"{BASE_URL}/invocations"

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
                    message = "The model was correct!"
                else:
                    message = "The model did not deliver the correct class! Maybe next time!"
        except Exception as e:
            message = f"Error communicating with the model server: {str(e)}"

    return render_template("index.html", categories=CATEGORIES, prediction=prediction, message=message)

# Frontend template
TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wine Quality Prediction</title>
</head>
<body>
    <h1>Wine Quality Prediction</h1>
    <form method="POST">
        <h3>Input Features</h3>
        {% for category in categories %}
            <label for="{{ category }}">{{ category }}</label>
            <input type="number" step="any" name="{{ category }}" required><br>
        {% endfor %}

        <h3>Actual Class (Optional)</h3>
        <label for="actual_class">Actual Wine Quality Class</label>
        <input type="number" name="actual_class"><br>

        <button type="submit">Submit</button>
    </form>

    {% if prediction is not none %}
        <h3>Prediction</h3>
        <p>Predicted Wine Quality Class: {{ prediction }}</p>
    {% endif %}

    {% if message %}
        <h3>Message</h3>
        <p>{{ message }}</p>
    {% endif %}
</body>
</html>
"""

@app.route("/template")
def template():
    return TEMPLATE

if __name__ == "__main__":
    app.run(debug=True)

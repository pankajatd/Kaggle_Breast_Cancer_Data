from flask import Flask, request, render_template
import pickle
import numpy as np
import os

import webbrowser
from threading import Timer

app = Flask(__name__)

# Load the model from the same folder as app.py
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        # Read all 10 features from the form
        features = [
            float(request.form["radius_mean"]),
            float(request.form["texture_mean"]),
            float(request.form["perimeter_mean"]),
            float(request.form["area_mean"]),
            float(request.form["smoothness_mean"]),
            float(request.form["compactness_mean"]),
            float(request.form["concavity_mean"]),
            float(request.form["concave_points_mean"]),
            float(request.form["symmetry_mean"]),
            float(request.form["fractal_dimension_mean"])
        ]

        # Convert to 2D array
        final_features = np.array([features])

        # Make prediction
        prediction = model.predict(final_features)[0]
        result = "Malignant (Cancer Detected)" if prediction == 1 else "Benign (No Cancer)"

        return render_template('home.html', prediction_text=f"Prediction Result: {result}")

    except Exception as e:
        return render_template('home.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    # Open the browser after 1 second to ensure server is running
    Timer(1, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=True)

import numpy as np
from flask import Flask, request, jsonify, render_template
from fastai.vision.all import *
from PIL import Image
import io
import torch
import base64

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Create flask app
flask_app = Flask(__name__)

# Load the model
model = load_learner("U:\downloads\ML-MODEL-DEPLOYMENT-USING-FLASK-main\ML-MODEL-DEPLOYMENT-USING-FLASK-main\eyedisease.pkl", "eyedisease.pkl")
severity_model = load_learner("U:\downloads\ML-MODEL-DEPLOYMENT-USING-FLASK-main\ML-MODEL-DEPLOYMENT-USING-FLASK-main\dr.pkl", "dr.pkl")

@flask_app.route("/")
def home():
    return render_template("safeindex.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    file = request.files["pre_image"]  # Get the uploaded file
    image = PILImage.create(file)  # Create a fastai PILImage from the file

    # Perform inference using your Learner
    prediction, _, probs = model.predict(image)

    # Convert the PIL image to base64 string for rendering in HTML
    prediction_severity=" "
    image_data = io.BytesIO()
    image.save(image_data, format='PNG')
    image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')
    if prediction=="diabetic_retinopathy":
        prediction_severity, _, probs = severity_model.predict(image)

    return render_template("safeindex.html", prediction_text=str(prediction), prediction_severity=str(prediction_severity),image_base64=image_base64)

@flask_app.route("/takephoto")
def takephoto():
    return render_template("takephoto.html")

# @flask_app.route("/severity", methods=["POST"])
# def predict_severity():
#     # file = request.files["image"]  # Get the uploaded file
#     # image = PILImage.create(file)  # Create a fastai PILImage from the file

#     # Perform inference using your Learner
#     prediction, _, probs = severity_model.predict(image)

#     # Convert the PIL image to base64 string for rendering in HTML
#     image_data = io.BytesIO()
#     image.save(image_data, format='PNG')
#     image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')

#     return render_template("safeindex.html", prediction_severity=str(prediction), image_base64=image_base64)

if __name__ == "__main__":
    flask_app.run(debug=True)

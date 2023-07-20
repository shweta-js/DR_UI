import numpy as np
from flask import Flask, request, jsonify, render_template,Response,session,send_file
from fastai.vision.all import *
from PIL import Image
import io
import torch
import base64
from flask_mail import Mail,Message
import os
import base64
import cv2

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Create flask app
flask_app = Flask(__name__)
flask_app.secret_key = 'your_secret_key'

# Load the model
model = load_learner("U:\downloads\ML-MODEL-DEPLOYMENT-USING-FLASK-main\ML-MODEL-DEPLOYMENT-USING-FLASK-main\eyedisease.pkl", "eyedisease.pkl")
severity_model = load_learner("U:\downloads\ML-MODEL-DEPLOYMENT-USING-FLASK-main\ML-MODEL-DEPLOYMENT-USING-FLASK-main\dr.pkl", "dr.pkl")

captured_image = None

@flask_app.route("/")
def home():
    return render_template("index.html")

@flask_app.route("/proceed")
def proceed():
    return render_template("upload.html")

@flask_app.route("/troubleshooting")
def troubleshooting():
    return render_template("troubleshooting.html")

@flask_app.route("/faq")
def faq():
    return render_template("faq.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    global prediction_value
    global g_image
    file = request.files["pre_image"]  # Get the uploaded file
    image = PILImage.create(file)  # Create a fastai PILImage from the file
    
    # Perform inference using your Learner
    prediction, _, probs = model.predict(image)
    prediction_value=prediction
    # Convert the PIL image to base64 string for rendering in HTML
    prediction_severity=" "
    image_data = io.BytesIO()
    image.save(image_data, format='PNG')
    image_base64 = base64.b64encode(image_data.getvalue()).decode('utf-8')
    g_image=image_base64
    if prediction=="diabetic_retinopathy":
        prediction_severity, _, probs = severity_model.predict(image)
    else:
        prediction_severity="."
    #for downloading pdf
    session['prediction_value'] = prediction_value
    session['image_base64'] = image_base64

    return render_template("upload.html", prediction_text=str(prediction), prediction_severity=str(prediction_severity),image_base64=image_base64)


@flask_app.route("/takephoto")
def takephoto():
    return render_template("takephoto.html")

@flask_app.route("/about")
def about():
    return render_template("about.html")

flask_app.config['MAIL_SERVER'] = 'smtp.gmail.com'
flask_app.config['MAIL_PORT'] = 465  # or your mail server port
flask_app.config['MAIL_USE_TLS'] = False  # if applicable
flask_app.config['MAIL_USERNAME'] = 'shwetajs.lrn@gmail.com'
flask_app.config['MAIL_PASSWORD'] = 'oeiaticuaycosoby'
flask_app.config['MAIL_USE_SSL'] = True

mail = Mail(flask_app)


@flask_app.route("/handle_report", methods=["POST"])
def handle_report():
    action = request.form.get("action")

    if action == "download":
        # Call the download_report function
        return download_report()
    elif action == "email":
        # Call the send_email function
        return send_email()
    else:
        # Handle other actions or show an error
        return "Invalid action"


@flask_app.route("/send_email", methods=["POST"])
def send_email():
    email = request.form.get("email")
    name = request.form.get("name")
    result = request.form.get("result")

    # Code to send email using Flask-Mail
    if request.method == 'POST':
        msg = Message("Report of your recent eye examination", 
                    sender=flask_app.config['MAIL_USERNAME'], 
                    recipients=[email])
        msg.body = f"Name: {name}\nResult: {prediction_value}"
        msg.attach("image.png", "image/png", base64.b64decode(g_image))
        mail.send(msg)
        return "Email sent successfully"

@flask_app.route("/download_report", methods=["GET"])
def download_report():
    # Retrieve the prediction value and image base64 from session or any other source
    prediction_value = prediction_value
    image_base64 = g_image  # Replace with the actual base64 string of the image

    # Generate HTML for the report using a template
    html = render_template("report_template.html", prediction_value=prediction_value)

    # Embed the image in the HTML template
    html_with_image = embed_image_in_html(html, image_base64)

    # Define the paths for the HTML and PDF files
    html_file_path = "report.html"
    pdf_file_path = "report.pdf"

    # Save the HTML to a file
    with open(html_file_path, "w") as f:
        f.write(html_with_image)

    # Generate the PDF from the HTML file
    options = {
        'enable-local-file-access': None  # Enable access to local files (e.g., image file)
    }
    pdfkit.from_file(html_file_path, pdf_file_path, options=options)

    # Remove the temporary HTML file
    os.remove(html_file_path)

    # Create a response with the PDF file
    response = make_response(send_file(pdf_file_path, as_attachment=True, attachment_filename="report.pdf"))

    # Set the Content-Disposition header for downloading
    response.headers["Content-Disposition"] = "attachment; filename=report.pdf"

    return response


def embed_image_in_html(html, image_base64):
    image_tag = f'<img src="data:image/png;base64,{image_base64}" alt="Image">'
    html_with_image = html.replace('<img-placeholder>', image_tag)
    return html_with_image


@flask_app.route('/capture', methods=['POST'])
def capture():
    global captured_image
    _, frame = cv2.VideoCapture(0).read()
    captured_image = frame

    # Convert the image array to PIL image
    image = Image.fromarray(captured_image)

    # Perform prediction
    prediction, _, _ = predict(image)


    return f"Image captured and predicted successfully! Prediction: {prediction}"

if __name__ == "__main__":
    flask_app.run(debug=True)
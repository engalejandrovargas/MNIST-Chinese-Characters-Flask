# Important imports
from app import app
from flask import request, render_template, url_for
import os
import cv2
import numpy as np
from PIL import Image
import random
import string
from app.model import preprocess_image, load_model, predict_image_class

# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'

# Load the trained model once when the app starts
loaded_model, device = load_model()

# Route to home page
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        full_filename = 'images/white_bg.jpg'
        return render_template("index.html", full_filename=full_filename)

    if request.method == "POST":
        image_upload = request.files['image_upload']
        imagename = image_upload.filename
        image = Image.open(image_upload)

        # Converting image to array
        image_arr = np.array(image.convert('RGB'))
        # Converting image to grayscale
        gray_img_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
        # Converting image back to RGB
        image = Image.fromarray(gray_img_arr)

        # Printing lowercase
        letters = string.ascii_lowercase
        # Generating unique image name for dynamic image display
        name = ''.join(random.choice(letters) for i in range(10)) + '.png'
        full_filename = 'uploads/' + name

        # Saving image to display in html
        img = Image.fromarray(image_arr, 'RGB')
        img.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name))

        # Preprocess the saved image for prediction
        image_path = os.path.join(app.config['INITIAL_FILE_UPLOADS'], name)
        image_tensor = preprocess_image(image_path)

        # Make predictions
        predicted_class = predict_image_class(loaded_model, device, image_tensor)

        # Returning template with filename and prediction
        return render_template('index.html', full_filename=full_filename, prediction=predicted_class)

# Main function
if __name__ == '__main__':
    app.run(debug=True)







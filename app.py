from flask import Flask, render_template, request, jsonify
import base64
import cv2
# import numpy as np
from processing.sketch import sketch
from processing.cartoonify import cartoonify
from processing.colorize import colorize
app = Flask(__name__)

# Define your image processing functions here
# def sketch(image_path):
#     # Your sketch function implementation
#     pass
# def sketch(img):
#     # Convert image to greyscale
#     gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
#     # Invert the greyscaled image
#     inverted_image = 255 - gray_image
    
#     # Smoothens the inverted image using GaussianBlur
#     blurred = cv2.GaussianBlur(inverted_image, (41, 1), 11)
    
#     # Again invert the blurred inverted image
#     inverted_blurred = 255 - blurred
    
#     # Combine gray image and inverted blurred image to form the final sketch image
#     pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
    
#     # Convert the resulting sketch image to base64
#     _, sketch_encoded = cv2.imencode('.png', pencil_sketch)
#     sketch_base64 = base64.b64encode(sketch_encoded).decode('utf-8')

#     return sketch_base64
# def cartoonify(image_path):
#     # Your cartoonify function implementation
#     pass

# def colorize(image_path):
#     # Your colorize function implementation
#     pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the image file and processing choice from the request
    image = request.files['image']
    choice = request.form['choice']

    # Save the uploaded image to a temporary file
    image_path = 'static/temp_image.png'
    image.save(image_path)
    # img = cv2.imread(image_path)
    # Perform the selected image processing based on the user's choice
    if choice == 'sketch':
        result_path = sketch(image_path)
        # result_path = sketch(image)
    elif choice == 'cartoonify':
        result_path = cartoonify(image_path, 8)
    elif choice == 'colorize':
        result_path = colorize(image_path)
    else:
        return jsonify({'error': 'Invalid choice'})

    print("Result success",result_path)
    # Convert the result image to base64 for displaying in the frontend
    with open(result_path, 'rb') as result_file:
        result_base64 = base64.b64encode(result_file.read()).decode('utf-8')

    return jsonify({'result': result_base64})

if __name__ == '__main__':
    app.run(debug=True)

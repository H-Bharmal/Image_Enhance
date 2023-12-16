# from flask import Flask, render_template, request, jsonify
# import base64
# import os
# # import numpy as np
# from processing.sketch import sketch
# from processing.cartoonify import cartoonify
# from processing.colorize import colorize
# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/process_image', methods=['POST'])
# def process_image():
#     # Get the image file and processing choice from the request
#     image = request.files['image']
#     choice = request.form['choice']

#     # Save the uploaded image to a temporary file
#     # image_path = 'static/temp_image.png'
#     # image.save(image_path)
#     # Get the directory of the current script
#     script_dir = os.path.dirname(os.path.realpath(__file__))

#     # Construct the absolute path for the image
#     image_path = os.path.join(script_dir, 'static', 'temp_image.png')
    
#     # Save the uploaded image to the absolute path
#     image.save(image_path)
#     # img = cv2.imread(image_path)
#     # Perform the selected image processing based on the user's choice
#     if choice == 'sketch':
#         result_path = sketch(image_path)
#         # result_path = sketch(image)
#     elif choice == 'cartoonify':
#         result_path = cartoonify(image_path, 8)
#     elif choice == 'colorize':
#         result_path = colorize(image_path)
#     else:
#         return jsonify({'error': 'Invalid choice'})

#     print("Result success",result_path)
#     # Convert the result image to base64 for displaying in the frontend
#     with open(result_path, 'rb') as result_file:
#         result_base64 = base64.b64encode(result_file.read()).decode('utf-8')

#     return jsonify({'result': result_base64})

# if __name__ == '__main__':
#     app.run(debug=True)








# ---------------------------------TRY
from flask import Flask, render_template, request, jsonify
import base64
import os
from processing.sketch import sketch
from processing.cartoonify import cartoonify
from processing.colorize import colorize

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    # Get the image file and processing choice from the request
    image = request.files['image']
    choice = request.form['choice']

    # Perform the selected image processing based on the user's choice
    if choice == 'sketch':
        result_base64 = sketch(image)
    elif choice == 'cartoonify':
        result_base64 = cartoonify(image, 8)
    elif choice == 'colorize':
        result_base64 = colorize(image)
    else:
        return jsonify({'error': 'Invalid choice'})

    return jsonify({'result': result_base64})

if __name__ == '__main__':
    app.run(debug=True)

import cv2
import base64
from image_utils import rescale_image
import numpy as np

def sketch(image):
    if isinstance(image, str):  # If the input is a file path
        img = cv2.imread(image)
    else:  # If the input is an image object
        # Convert image to NumPy array
        nparr = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Rest of the processing remains the same
    # Convert image to greyscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert the greyscaled image
    inverted_image = 255 - gray_image
    # Smoothens the inverted image using GaussianBlur
    blurred = cv2.GaussianBlur(inverted_image, (41, 1), 11)
    # Again invert the blurred inverted image
    inverted_blurred = 255 - blurred
    # Combine gray image and inverted blurred image to form the final sketch image
    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)

    # Optionally, rescale the image
    # pencil_sketch = rescale_image(pencil_sketch, scale=0.5)

    # Convert the resulting sketch image to base64
    _, sketch_encoded = cv2.imencode('.png', pencil_sketch)
    sketch_base64 = base64.b64encode(sketch_encoded).decode('utf-8')

    return sketch_base64

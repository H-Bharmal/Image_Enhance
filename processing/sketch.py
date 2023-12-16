
# import cv2
# import numpy as np
# import base64

# def sketch(img):
#     # print(type(img))
#     # img_path ='./static/temp_image.png'
#     # img = cv2.imread(img_path)
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

import cv2
from image_utils import rescale_image
# from resize import fit_image_within_box
def sketch(img):
   
    img = cv2.imread(img)
    
    #convert image to greyscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("gray.png", gray_image)
    
    # inverts the  greyscaled image
    inverted_image = 255 - gray_image
    # cv2.imwrite("inv.png", inverted_image)
    
    # smoothens the inverted image, it makes it easier to analyse and see details.
    blurred = cv2.GaussianBlur(inverted_image, (41, 1), 11)
    # cv2.imwrite("blur.png", blurred)
    
    # Again inverts the blurred inverrteed image to nullify the effect
    inverted_blurred = 255 - blurred
    # cv2.imwrite("invblur.png", inverted_blurred)
    
    # Combines gray image and inverted blurred image to form final sketch image, i.e change data from singular column to multicolumn.
    pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
    
    #resize the image
    cv2.imwrite("Sketch.png", pencil_sketch)
    rescale_image("Sketch.png", "Sketch.png", 0.5)

    # screen_height = 2
    # output_height = int(pencil_sketch.shape[0] / screen_height)
    # output_width = int((output_height / pencil_sketch.shape[0]) * pencil_sketch.shape[1])
    # cv2.imwrite("Sketch.png", pencil_sketch)
    
    return "Sketch.png"
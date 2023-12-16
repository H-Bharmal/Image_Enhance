import cv2
import numpy as np
def fit_image_within_box(input_path, output_path, box_width, box_height):
    # Read the input image
    cartoon = cv2.imread(input_path)

    # Get the original dimensions of the image
    original_height, original_width = cartoon.shape[:2]

    # Calculate the aspect ratio of the original image
    aspect_ratio = original_width / original_height

    # Calculate the dimensions to fit the image within the specified box
    if box_width / aspect_ratio <= box_height:
        fit_width = int(box_width)
        fit_height = int(box_width / aspect_ratio)
    else:
        fit_width = int(box_height * aspect_ratio)
        fit_height = int(box_height)

    # Resize the image to fit within the specified box
    resized_image = cv2.resize(cartoon, (fit_width, fit_height))

    # Create a blank image with the specified box dimensions
    result_image = 255 * np.ones((box_height, box_width, 3), dtype=np.uint8)

    # Calculate the position to paste the resized image at the center
    paste_position = ((box_width - fit_width) // 2, (box_height - fit_height) // 2)

    # Paste the resized image onto the blank image at the calculated position
    result_image[paste_position[1]:paste_position[1] + fit_height,
                 paste_position[0]:paste_position[0] + fit_width] = resized_image

    # Save the result image
    cv2.imwrite(output_path, result_image)
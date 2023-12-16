from PIL import Image

def rescale_image(input_path, output_path, scale_factor):
    # Open the image file
    original_image = Image.open(input_path)

    # Get the original width and height
    original_width, original_height = original_image.size

    # Calculate the new width and height based on the scale factor
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image
    resized_image = original_image.resize((new_width, new_height))

    # Save the resized image
    resized_image.save(output_path)

# Example usage
# input_image_path = 'input.jpg'
# output_image_path = 'output_rescaled.jpg'
# scale_factor = 0.8  # Adjust this value as needed

# rescale_image(input_image_path, output_image_path, scale_factor)


# import cv2
# import numpy as np
# def fit_image_within_box(input_path, output_path, box_width, box_height):
#     # Read the input image
#     cartoon = cv2.imread(input_path)

#     # Get the original dimensions of the image
#     original_height, original_width = cartoon.shape[:2]

#     # Calculate the aspect ratio of the original image
#     aspect_ratio = original_width / original_height

#     # Calculate the dimensions to fit the image within the specified box
#     if box_width / aspect_ratio <= box_height:
#         fit_width = int(box_width)
#         fit_height = int(box_width / aspect_ratio)
#     else:
#         fit_width = int(box_height * aspect_ratio)
#         fit_height = int(box_height)

#     # Resize the image to fit within the specified box
#     resized_image = cv2.resize(cartoon, (fit_width, fit_height))

#     # Create a blank image with the specified box dimensions
#     result_image = 255 * np.ones((box_height, box_width, 3), dtype=np.uint8)

#     # Calculate the position to paste the resized image at the center
#     paste_position = ((box_width - fit_width) // 2, (box_height - fit_height) // 2)

#     # Paste the resized image onto the blank image at the calculated position
#     result_image[paste_position[1]:paste_position[1] + fit_height,
#                  paste_position[0]:paste_position[0] + fit_width] = resized_image

#     # Save the result image
#     cv2.imwrite(output_path, result_image)
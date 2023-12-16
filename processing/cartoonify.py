import cv2
import numpy as np

# def cartoonify(img, k):
#     # Convert the input image to gray scale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Perform adaptive threshold
#     edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)

#     # Defining input data for clustering
#     data = np.float32(img).reshape((-1, 3))

#     # Defining criteria
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

#     # Applying cv2.kmeans function
#     _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     center = np.uint8(center)

#     # Reshape the output data to the size of input image
#     result = center[label.flatten()]
#     result = result.reshape(img.shape)

#     # Smooth the result
#     blurred = cv2.medianBlur(result, 3)

#     # Combine the result and edges to get the final cartoon effect
#     cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

#     return cartoon

# import cv2
# import numpy as np
# import base64

# def cartoonify(img, k):
#     # Convert the input image to grayscale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Perform adaptive threshold
#     edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)

#     # Reshape the input data for clustering
#     data = np.float32(img).reshape((-1, 3))

#     # Define criteria
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

#     # Apply cv2.kmeans function
#     _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#     center = np.uint8(center)

#     # Reshape the output data to the size of the input image
#     result = center[label.flatten()]
#     result = result.reshape(img.shape)

#     # Smooth the result
#     blurred = cv2.medianBlur(result, 3)

#     # Combine the result and edges to get the final cartoon effect
#     cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

#     # Convert the resulting cartoon image to base64
#     _, cartoon_encoded = cv2.imencode('.png', cartoon)
#     cartoon_base64 = base64.b64encode(cartoon_encoded).decode('utf-8')

#     return cartoon_base64


import cv2
import numpy as np
def cartoonify(img, k):
    img = cv2.imread(img)
    # Convert the input image to gray scale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Peform adaptive threshold
    edges  = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)

    # cv2.imshow('edges', edges)

    # Defining input data for clustering
    data = np.float32(img).reshape((-1, 3))

    print("shape of input data: ", img.shape)
    print('shape of resized data', data.shape)

    # Defining criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # Applying cv2.kmeans function
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    # print(center)

    # Reshape the output data to the size of input image
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    #cv2.imshow("result", result)

    # Smooth the result
    blurred = cv2.medianBlur(result, 3)
    # Combine the result and edges to get final cartoon effect
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

    #resize the image
    screen_height = 2
    output_height = int(cartoon.shape[0] / screen_height)
    output_width = int((output_height / cartoon.shape[0]) * cartoon.shape[1])
    cv2.imwrite("cartoon.png", cartoon)

    return "cartoon.png"
import cv2
import numpy as np
import cv2
import numpy as np
import base64

def cartoonify(image, k):
    if isinstance(image, str):  # If the input is a file path
        img = cv2.imread(image)
    else:  # If the input is an image object
        # Convert image to NumPy array
        nparr = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the input image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform adaptive threshold
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 8)

    # Defining input data for clustering
    data = np.float32(img).reshape((-1, 3))

    # Defining criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # Applying cv2.kmeans function
    _, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)

    # Reshape the output data to the size of the input image
    result = center[label.flatten()]
    result = result.reshape(img.shape)

    # Smooth the result
    blurred = cv2.medianBlur(result, 3)

    # Combine the result and edges to get the final cartoon effect
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

    # Convert the resulting cartoon image to base64
    _, cartoon_encoded = cv2.imencode('.png', cartoon)
    cartoon_base64 = base64.b64encode(cartoon_encoded).decode('utf-8')

    return cartoon_base64

import cv2
import numpy as np
import base64
import os
import requests
from io import BytesIO
import urllib
import tempfile

def colorize(image):
    if isinstance(image, str):  # If the input is a file path
        img = cv2.imread(image)
    else:  # If the input is an image object
        # Convert image to NumPy array
        nparr = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Get the absolute path of the current script
    # script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct relative paths to your external files
    # model_path = os.path.join(script_dir, 'colorization_release_v2.caffemodel')
    # model_path = 'D:\Learn\College Projects\Colorize_Final\processing\colorization_release_v2.caffemodel'

    script_dir = os.path.dirname(__file__)  # Assuming this script is in the same directory as the model files

    model_path = os.path.join(script_dir, 'colorization_release_v2.caffemodel')
    protxt_path = os.path.join(script_dir, 'colorization_deploy_v2.prototxt')
    points_path = os.path.join(script_dir, 'pts_in_hull.npy')
    image_path = os.path.join(script_dir, 'static', 'temp_image.png')

    # Download links
    download_Link_Model = "https://drive.google.com/uc?export=download&id=1AoPskjkS7HdNk7tXmK6b7KBr1Yh2yTEf&confirm=t&uuid=f556b515-3a43-4e22-8f61-d6ac50bc31d7"
    download_Link_Points = "https://drive.google.com/u/0/uc?id=1uNXx3MfFj-NBhm2QAk_A5IvnLC0i41wx&export=download"

    def download_file_from_url(url, destination):
        response = requests.get(url, stream=True)
        with open(destination, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)

    def load_numpy_array_from_url(url):
        response = requests.get(url)
        npy_data = BytesIO(response.content)
        array_data = np.load(npy_data)
        return array_data

    # Download the model
    download_file_from_url(download_Link_Model, model_path)

    # Load the model and points
    pts = load_numpy_array_from_url(download_Link_Points)
    net = cv2.dnn.readNetFromCaffe(protxt_path, model_path)
    # Load the model and points
    # net = load_model_from_url(protxt_path, download_Link_Model)
    # # net = load_model_from_url(download_Link_Model)
    # pts = load_numpy_array_from_url(download_Link_Points)

    # net = cv2.dnn.readNetFromCaffe(protxt_path, model_path)
    # ab channel - 1x1 convolutions and add them to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId('conv8_313_rh')
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype('float32')]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]

    # Scale and convert to float type with the dnn model
    # Image converted to LAB format
    scaled = img.astype('float32') / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resizing the image
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # L channel
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Convert the resulting colorized image to base64
    _, colorized_encoded = cv2.imencode('.png', colorized)
    colorized_base64 = base64.b64encode(colorized_encoded).decode('utf-8')

    return colorized_base64

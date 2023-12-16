import numpy as np
import cv2
import os
def colorize(image):

    # Get the absolute path of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Construct relative paths to your external files
    model = os.path.join(script_dir, 'colorization_release_v2.caffemodel')
    protxt = os.path.join(script_dir, 'colorization_deploy_v2.prototxt')
    points = os.path.join(script_dir, 'pts_in_hull.npy')
    image_path = os.path.join(script_dir, 'static', 'temp_image.png')


    # image_path = './tree.jpg'
    # model = 'https://drive.google.com/file/d/1AoPskjkS7HdNk7tXmK6b7KBr1Yh2yTEf/view?usp=sharing'
    download_Link_Model = "https://drive.google.com/uc?export=download&id=1AoPskjkS7HdNk7tXmK6b7KBr1Yh2yTEf&confirm=t&uuid=f556b515-3a43-4e22-8f61-d6ac50bc31d7"
    
    download_Link_Points = "https://drive.google.com/u/0/uc?id=1uNXx3MfFj-NBhm2QAk_A5IvnLC0i41wx&export=download"
    import urllib
    from io import BytesIO
    import tempfile

    def load_model_from_url(url):
        response = urllib.request.urlopen(url)
        proto_model_data = response.read()

        # Save the data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(proto_model_data)
            temp_file_path = temp_file.name

        # Load the model from the temporary file
        net = cv2.dnn.readNetFromCaffe(temp_file_path)

        # Remove the temporary file
        temp_file.close()
        # Uncomment the line below if you want to keep the temporary file
        # os.unlink(temp_file_path)

        return net
    import requests
    def load_numpy_array_from_url(url):
        response = requests.get(url)
        npy_data = BytesIO(response.content)
        array_data = np.load(npy_data)
        return array_data
    # def load_model_from_url(url):
    #     response = urllib.request.urlopen(url)
    #     proto_model = BytesIO(response.read())
    #     net = cv2.dnn.readNetFromCaffe(proto_model)
    #     return net
    # # protxt = './colorization_deploy_v2.prototxt'
    # model = 'D:\Learn\College Projects\Colorize_Final\processing\colorization_release_v2.caffemodel'
    # protxt = 'D:\Learn\College Projects\Colorize_Final\processing\colorization_deploy_v2.prototxt'
    # points = 'D:\Learn\College Projects\Colorize_Final\processing\pts_in_hull.npy'
    image_path = 'static/temp_image.png'

    #To load the model from internet
    # net = load_model_from_url(download_Link_Model) #TODO : make it accessible from anywhere
    # pts = load_numpy_array_from_url(download_Link_Points)

    #for offline way
    net = cv2.dnn.readNetFromCaffe(protxt, model)
    pts = np.load(points)

    # ab channel - 1x1 convolutions and add them to model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId('conv8_313_rh')
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype('float32')]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]

    # Input the image
    # Scale and convert to float type with the dnn model
    # Image converted to LAB format
    image = cv2.imread(image_path)
    scaled = image.astype('float32') / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resizing the image
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # L channel
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # Resize the output images to half the view height of the screen
    screen_height = 2  # Change this value according to your desired scale
    output_height = int(image.shape[0] / screen_height)
    output_width = int((output_height / image.shape[0]) * image.shape[1])
    colorized_resized = cv2.resize(colorized, (output_width, output_height))

    image_resized = cv2.resize(image, (output_width, output_height))

    # Original and Colorized image
    # cv2.imshow("Original Image", image_resized)
    # cv2.imshow("Colorized Image", colorized_resized)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("colorized.jpg",colorized_resized)
    return "colorized.jpg"

# import cv2
# import numpy as np
# import base64

# def colorize(image):
#     image = cv2.imread(image)
#     # protxt = 'colorization_deploy_v2.prototxt'
#     model = 'D:\Learn\College Projects\Colorize_Final\processing\colorization_release_v2.caffemodel'
#     protxt = 'D:\Learn\College Projects\Colorize_Final\processing\colorization_deploy_v2.prototxt'
#     # model = 'Colorize_Final\colorization_release_v2.caffemodel'
#     points = 'D:\Learn\College Projects\Colorize_Final\processing\pts_in_hull.npy'
#     # points = 'Colorize_Final\pts_in_hull.npy'

#     net = cv2.dnn.readNetFromCaffe(protxt, model)
#     pts = np.load(points)

#     # ab channel - 1x1 convolutions and add them to model
#     class8 = net.getLayerId("class8_ab")
#     conv8 = net.getLayerId('conv8_313_rh')
#     pts = pts.transpose().reshape(2, 313, 1, 1)
#     net.getLayer(class8).blobs = [pts.astype('float32')]
#     net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype='float32')]

#     # Scale and convert to float type with the dnn model
#     # Image converted to LAB format
#     scaled = image.astype('float32') / 255.0
#     lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

#     # Resizing the image
#     resized = cv2.resize(lab, (224, 224))
#     L = cv2.split(resized)[0]
#     L -= 50

#     # L channel
#     net.setInput(cv2.dnn.blobFromImage(L))
#     ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
#     ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

#     L = cv2.split(lab)[0]
#     colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
#     colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
#     colorized = np.clip(colorized, 0, 1)
#     colorized = (255 * colorized).astype("uint8")

#     # Convert the resulting colorized image to base64
#     _, colorized_encoded = cv2.imencode('.png', colorized)
#     colorized_base64 = base64.b64encode(colorized_encoded).decode('utf-8')

#     return colorized_base64

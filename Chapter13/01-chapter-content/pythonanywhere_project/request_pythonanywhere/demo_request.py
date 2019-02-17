"""
 Request example to perform a POST request in order to detect and draw the faces in a image using the
 API hosted at pythonanywhere
"""


# Import required packages:
import cv2
import numpy as np
import requests
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


FACE_DETECTION_REST_API_URL = "http://opencv.pythonanywhere.com/detect"
IMAGE_PATH = "test_face_processing.jpg"

# Load the image and construct the payload:
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# Submit the POST request:
r = requests.post(FACE_DETECTION_REST_API_URL, files=payload)

# See the response:
print("status code: {}".format(r.status_code))
print("headers: {}".format(r.headers))
print("content: {}".format(r.json()))

# Get JSON data from the response and get 'result':
json_data = r.json()
result = json_data['result']

# Convert the loaded image to the OpenCV format:
image_array = np.asarray(bytearray(image), dtype=np.uint8)
img_opencv = cv2.imdecode(image_array, -1)

# Draw faces in the OpenCV image:
for face in result:
    left, top, right, bottom = face['box']
    # To draw a rectangle, you need top-left corner and bottom-right corner of rectangle:
    cv2.rectangle(img_opencv, (left, top), (right, bottom), (0, 255, 255), 2)
    # Draw top-left corner and bottom-right corner (checking):
    cv2.circle(img_opencv, (left, top), 5, (0, 0, 255), -1)
    cv2.circle(img_opencv, (right, bottom), 5, (255, 0, 0), -1)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(8, 6))
plt.suptitle("Using face API at http://opencv.pythonanywhere.com/detect", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Show the output image
show_img_with_matplotlib(img_opencv, "face detection", 1)

# Show the Figure:
plt.show()
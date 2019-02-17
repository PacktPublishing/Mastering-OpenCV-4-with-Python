"""
 Request example to perform a POST request using the Keras Deep Learning REST API and also drawing the results
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


KERAS_REST_API_URL = "http://localhost:5000/predict"
IMAGE_PATH = "car.jpg"

# Load the image and construct the payload:
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

# Submit the POST request:
r = requests.post(KERAS_REST_API_URL, files=payload).json()

# Convert the loaded image to the OpenCV format:
image_array = np.asarray(bytearray(image), dtype=np.uint8)
img_opencv = cv2.imdecode(image_array, -1)
img_opencv = cv2.resize(img_opencv, (500, 500))

y_pos = 40

# Show the results:
if r["success"]:
    # Iterate over the predictions
    for (i, result) in enumerate(r["predictions"]):
        # Print the results:
        print("{}. {}: {:.4f}".format(i + 1, result["label"], result["probability"]))
        # Render the results in the image:
        cv2.putText(img_opencv, "{}. {}: {:.4f}".format(i + 1, result["label"], result["probability"]),
                    (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        y_pos += 30
else:
    print("Request failed")

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(8, 6))
plt.suptitle("Using Keras Deep Learning REST API", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Show the output image
show_img_with_matplotlib(img_opencv, "Classification results (NASNetMobile)", 1)

# Show the Figure:
plt.show()

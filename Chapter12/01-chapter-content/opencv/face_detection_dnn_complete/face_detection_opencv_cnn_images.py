"""
Face detection using OpenCV DNN face detector when feeding several images to the network
"""

# Import required packages:
import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Load pre-trained model:
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Load images and get the list of images:
image = cv2.imread("face_test.jpg")
image2 = cv2.imread("face_test_2.jpg")
images = [image.copy(), image2.copy()]

# Call cv2.dnn.blobFromImages():
blob_images = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, False)

# Set the blob as input and obtain the detections:
net.setInput(blob_images)
detections = net.forward()

# Iterate over all detections:
# We have to check the first element of each detection to know which image it belongs to:
for i in range(0, detections.shape[2]):
    # First, we have to get the image the detection belongs to:
    img_id = int(detections[0, 0, i, 0])
    # Get the confidence of this prediction:
    confidence = detections[0, 0, i, 2]

    # Filter out weak predictions:
    if confidence > 0.25:
        # Get the size of the current image:
        (h, w) = images[img_id].shape[:2]

        # Get the (x,y) coordinates of the detection:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw bounding box and probability:
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(images[img_id], (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(images[img_id], text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(16, 8))
plt.suptitle("OpenCV DNN face detector when feeding several images", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Show the input and the output images with the detections:
show_img_with_matplotlib(image, "input img 1", 1)
show_img_with_matplotlib(image2, "input img 2", 2)
show_img_with_matplotlib(images[0], "output img 1", 3)
show_img_with_matplotlib(images[1], "output img 2", 4)

# Show the Figure:
plt.show()

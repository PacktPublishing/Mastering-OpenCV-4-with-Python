"""
Face detection using OpenCV DNN face detector when feeding several images to the network
and crop=True in function cv2.dnn.blobFromImages()
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


def get_cropped_imgs(imgs):
    """Returns the cropped images"""

    imgs_cropped = []

    for img in imgs:
        # Create a copy of the image:
        img_copy = img.copy()

        # calculate size of resulting image:
        size = min(img_copy.shape[1], img_copy.shape[0])

        # calculate x1, and y1
        x1 = int(0.5 * (img_copy.shape[1] - size))
        y1 = int(0.5 * (img_copy.shape[0] - size))

        # crop and return the image
        imgs_cropped.append(img_copy[y1:(y1 + size), x1:(x1 + size)])

    return imgs_cropped


# Load pre-trained model:
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Load images and get the list of images:
image = cv2.imread("face_test.jpg")
image2 = cv2.imread("face_test_2.jpg")
images = [image, image2]

# Get the cropped (original) images:
images_cropped = get_cropped_imgs(images)

# Call cv2.dnn.blobFromImages():
blob_cropped = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, True)

# Set the blob as input and obtain the detections:
net.setInput(blob_cropped)
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
        (h, w) = images_cropped[img_id].shape[:2]

        # Get the (x,y) coordinates of the detection:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw bounding box and probability:
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        # Note here that we are showing the results in the cropped images!:
        cv2.rectangle(images_cropped[img_id], (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(images_cropped[img_id], text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(16, 8))
plt.suptitle("OpenCV DNN face detector when feeding several images and cropping", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Show the input and the output images with the detections:
show_img_with_matplotlib(image, "input img 1", 1)
show_img_with_matplotlib(image2, "input img 2", 2)
show_img_with_matplotlib(images_cropped[0], "output cropped img 1", 3)
show_img_with_matplotlib(images_cropped[1], "output cropped img 2", 4)

# Show the Figure:
plt.show()

"""
Feature detection with ORB keypoint detector
"""

# Import required packages:
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 5))
plt.suptitle("ORB keypoint detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load test image:
image = cv2.imread('opencv_logo_with_text.png')

# Initiate ORB detector:
orb = cv2.ORB_create()

# Detect the keypoints using ORB:
keypoints = orb.detect(image, None)

# Compute the descriptors of the detected keypoints:
keypoints, descriptors = orb.compute(image, keypoints)

# Print one ORB descriptor:
print("First extracted descriptor: {}".format(descriptors[0]))

# Draw detected keypoints:
image_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 255), flags=0)

# Plot the images:
show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(image_keypoints, "detected keypoints", 2)

# Show the Figure:
plt.show()

"""
Showing images with both OpenCV and Matplotlib
"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Load image using cv2.imread:
img_OpenCV = cv2.imread('logo.png')

# Split the loaded image into its three channels (b, g, r):
b, g, r = cv2.split(img_OpenCV)

# Merge again the three channels but in the RGB format:
img_matplotlib = cv2.merge([r, g, b])

# Show both images (img_OpenCV and img_matplotlib) using matplotlib
# This will show the image in wrong color:
plt.subplot(121)
plt.imshow(img_OpenCV)
plt.title('img OpenCV')
# This will show the image in true color:
plt.subplot(122)
plt.imshow(img_matplotlib)
plt.title('img matplotlib')
plt.show()

# Show both images (img_OpenCV and img_matplotlib) using cv2.imshow()
# This will show the image in true color:
cv2.imshow('bgr image', img_OpenCV)
# This will show the image in wrong color:
cv2.imshow('rgb image', img_matplotlib)
cv2.waitKey(0)
cv2.destroyAllWindows()

# To stack horizontally (img_OpenCV to the left of img_matplotlib):
img_concats = np.concatenate((img_OpenCV, img_matplotlib), axis=1)

# Now, we show the concatenated image:
cv2.imshow('bgr image and rgb image', img_concats)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Using numpy capabilities to get the channels and two build the RGB image
# Get the three channels (instead of using cv2.split):
B = img_OpenCV[:, :, 0]
G = img_OpenCV[:, :, 1]
R = img_OpenCV[:, :, 2]

# Transform the image BGR to RGB using Numpy capabilities:
img_RGB = img_OpenCV[:, :, ::-1]

# Now, we show the RGB image:
cv2.imshow('img RGB (wrong color)', img_RGB)
cv2.waitKey(0)
cv2.destroyAllWindows()

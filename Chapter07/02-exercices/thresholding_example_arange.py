"""
Simple thresholding applied to a real image using np.arange() to create the different threshold values
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title and color:
fig = plt.figure(figsize=(9, 9))
plt.suptitle("Thresholding example", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
image = cv2.imread('sudoku.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Plot the grayscale images and the histograms:
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "img", 1)

# Get the array with the values for thresholding in the range [60-130] with step 10:
# This function returns an array of evenly spaced values:
threshold_values = np.arange(start=60, stop=140, step=10)
# print(threshold_values)

# Apply cv2.threshold() with the different threshold values defined in 'threshold_values'
thresholded_images = []
for threshold in threshold_values:
    ret, thresh = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    thresholded_images.append(thresh)

# Show the thresholded images:
for index, (thresholded_image, threshold_value) in enumerate(zip(thresholded_images, threshold_values)):
    show_img_with_matplotlib(cv2.cvtColor(thresholded_image, cv2.COLOR_GRAY2BGR), "threshold = " + str(threshold_value),
                             index + 2)

# Show the Figure:
plt.show()

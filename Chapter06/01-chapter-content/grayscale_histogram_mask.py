"""
Grayscale histograms using a mask
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_hist_with_matplotlib_gray(hist, title, pos, color):
    """Shows the histogram using matplotlib capabilities"""

    ax = plt.subplot(2, 2, pos)
    # plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


# Create the dimensions of the figure and set title:
plt.figure(figsize=(10, 6))
plt.suptitle("Grayscale masked histogram", fontsize=14, fontweight='bold')

# Load the image and convert it to grayscale:
image = cv2.imread('lenna_mod.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the histogram calling cv2.calcHist()
# The first argument it the list of images to process
# The second argument is the indexes of the channels to be used to calculate the histogram
# The third argument is a mask to compute the histogram for the masked pixels
# The fourth argument is a list containing the number of bins for each channel
# The fifth argument is the range of possible pixel values
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Plot the grayscale image and the histogram:
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 2, 'm')

# Create the mask and calculate the histogram using the mask:
mask = np.zeros(gray_image.shape[:2], np.uint8)
mask[30:190, 30:190] = 255
hist_mask = cv2.calcHist([gray_image], [0], mask, [256], [0, 256])

# Create the 'masked_img' (only for visualization) and show the grayscale masked histogram:
masked_img = cv2.bitwise_and(gray_image, gray_image, mask=mask)
show_img_with_matplotlib(cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR), "masked gray image", 3)
show_hist_with_matplotlib_gray(hist_mask, "grayscale masked histogram", 4, 'm')

# Show the Figure:
plt.show()

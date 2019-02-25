"""
Introduction to grayscale histogram equalization
"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_hist_with_matplotlib_gray(hist, title, pos, color):
    """Shows the histogram using matplotlib capabilities"""

    ax = plt.subplot(3, 4, pos)
    # plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


# Create the dimensions of the figure and set title:
plt.figure(figsize=(18, 14))
plt.suptitle("Grayscale histogram equalization with cv2.equalizeHist()", fontsize=16, fontweight='bold')

# Load the image and convert it to grayscale:
image = cv2.imread('lenna.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate the histogram calling cv2.calcHist()
# The first argument it the list of images to process
# The second argument is the indexes of the channels to be used to calculate the histogram
# The third argument is a mask to compute the histogram for the masked pixels
# The fourth argument is a list containing the number of bins for each channel
# The fifth argument is the range of possible pixel values
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Equalize image and calculate histogram:
gray_image_eq = cv2.equalizeHist(gray_image)
hist_eq = cv2.calcHist([gray_image_eq], [0], None, [256], [0, 256])

# Add 35 to every pixel on the grayscale image (the result will look lighter) and calculate histogram:
M = np.ones(gray_image.shape, dtype="uint8") * 35
added_image = cv2.add(gray_image, M)
hist_added_image = cv2.calcHist([added_image], [0], None, [256], [0, 256])

# Equalize image and calculate histogram:
added_image_eq = cv2.equalizeHist(added_image)
hist_eq_added_image = cv2.calcHist([added_image_eq], [0], None, [256], [0, 256])

# Subtract 35 from every pixel (the result will look darker) and calculate histogram:
subtracted_image = cv2.subtract(gray_image, M)
hist_subtracted_image = cv2.calcHist([subtracted_image], [0], None, [256], [0, 256])

# Equalize image and calculate histogram:
subtracted_image_eq = cv2.equalizeHist(subtracted_image)
hist_eq_subtracted_image = cv2.calcHist([subtracted_image_eq], [0], None, [256], [0, 256])

# Plot the images and the histograms (without equalization first):
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 2, 'm')
show_img_with_matplotlib(cv2.cvtColor(added_image, cv2.COLOR_GRAY2BGR), "gray lighter", 5)
show_hist_with_matplotlib_gray(hist_added_image, "grayscale histogram", 6, 'm')
show_img_with_matplotlib(cv2.cvtColor(subtracted_image, cv2.COLOR_GRAY2BGR), "gray darker", 9)
show_hist_with_matplotlib_gray(hist_subtracted_image, "grayscale histogram", 10, 'm')

# Plot the images and the histograms (with equalization):
show_img_with_matplotlib(cv2.cvtColor(gray_image_eq, cv2.COLOR_GRAY2BGR), "grayscale equalized", 3)
show_hist_with_matplotlib_gray(hist_eq, "grayscale equalized histogram", 4, 'm')
show_img_with_matplotlib(cv2.cvtColor(added_image_eq, cv2.COLOR_GRAY2BGR), "gray lighter equalized", 7)
show_hist_with_matplotlib_gray(hist_eq_added_image, "grayscale equalized histogram", 8, 'm')
show_img_with_matplotlib(cv2.cvtColor(subtracted_image_eq, cv2.COLOR_GRAY2BGR), "gray darker equalized", 11)
show_hist_with_matplotlib_gray(hist_eq_subtracted_image, "grayscale equalized histogram", 12, 'm')

# Show the Figure:
plt.show()

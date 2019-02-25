"""
Introduction to grayscale histograms and calculate brightness for every image
"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_hist_with_matplotlib_gray(hist, title, pos, color):
    """Shows the histogram using matplotlib capabilities"""

    ax = plt.subplot(2, 3, pos)
    # plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


def get_brightness(img):
    """Calculates the brightness of the image"""

    brightness = np.mean(img)
    return brightness


# Create the dimensions of the figure and set title:
plt.figure(figsize=(15, 6))
plt.suptitle("Grayscale histograms and brightness", fontsize=14, fontweight='bold')

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
# print("histogram shape: '{}'".format(hist.shape))

# Add 35 to every pixel on the grayscale image (the result will look lighter) and calculate histogram
M = np.ones(gray_image.shape, dtype="uint8") * 35
added_image = cv2.add(gray_image, M)
hist_added_image = cv2.calcHist([added_image], [0], None, [256], [0, 256])

# Subtract 35 from every pixel (the result will look darker) and calculate histogram
subtracted_image = cv2.subtract(gray_image, M)
hist_subtracted_image = cv2.calcHist([subtracted_image], [0], None, [256], [0, 256])

# Exercise: calculate the brightness of the images:
brightness_1 = get_brightness(gray_image)
brightness_2 = get_brightness(added_image)
brightness_3 = get_brightness(subtracted_image)

# Write these images to disk to be used for histogram comparison (see exercise 'compare_histogram.py')
# cv2.imwrite("gray_image.png", gray_image)
# cv2.imwrite("gray_added_image.png", added_image)
# cv2.imwrite("gray_subtracted_image.png", subtracted_image)
# cv2.imwrite("gray_blurred.png", cv2.blur(gray_image, (10, 10)))


# Plot the images and the histograms:
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR),
                         "gray" + str(' brightness = % 3.1f' % brightness_1), 1)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 4, 'm')
show_img_with_matplotlib(cv2.cvtColor(added_image, cv2.COLOR_GRAY2BGR),
                         "gray lighter" + str(' brightness = % 3.1f' % brightness_2), 2)
show_hist_with_matplotlib_gray(hist_added_image, "grayscale histogram", 5, 'm')
show_img_with_matplotlib(cv2.cvtColor(subtracted_image, cv2.COLOR_GRAY2BGR),
                         "gray darker" + str(' brightness = % 3.1f' % brightness_3), 3)
show_hist_with_matplotlib_gray(hist_subtracted_image, "grayscale histogram", 6, 'm')

# Show the Figure:
plt.show()

"""
Color histogram equalization using the HSV color space
"""

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


def show_hist_with_matplotlib_rgb(hist, title, pos, color):
    """Shows the histogram using matplotlib capabilities"""

    ax = plt.subplot(3, 4, pos)
    # plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])

    for (h, c) in zip(hist, color):
        plt.plot(h, color=c)


def hist_color_img(img):
    """Calculates the histogram for a three-channel image"""

    histr = []
    histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))
    histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))
    return histr


def equalize_hist_color_hsv(img):
    """Equalizes the image splitting it after HSV conversion and applying cv2.equalizeHist()
    to the V channel, merging the channels and convert back to the BGR color space
    """

    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image


# Create the dimensions of the figure and set title:
plt.figure(figsize=(18, 14))
plt.suptitle("Color histogram equalization with cv2.calcHist() in the V channel", fontsize=14, fontweight='bold')

# Load the original image and convert it to grayscale
image = cv2.imread('lenna.png')

# Calculate the histogram for this BGR image:
hist_color = hist_color_img(image)

# Equalize the image and calculate histogram:
image_eq = equalize_hist_color_hsv(image)
hist_image_eq = hist_color_img(image_eq)

# Add 15 to every pixel on the grayscale image (the result will look lighter) and calculate histogram
M = np.ones(image.shape, dtype="uint8") * 15
added_image = cv2.add(image, M)
hist_color_added_image = hist_color_img(added_image)

# Equalize image and calculate histogram
added_image_eq = equalize_hist_color_hsv(added_image)
hist_added_image_eq = hist_color_img(added_image_eq)

# Subtract 15 from every pixel (the result will look darker) and calculate histogram
subtracted_image = cv2.subtract(image, M)
hist_color_subtracted_image = hist_color_img(subtracted_image)

# Equalize image and calculate histogram
subtracted_image_eq = equalize_hist_color_hsv(subtracted_image)
hist_subtracted_image_eq = hist_color_img(subtracted_image_eq)

# Plot the images and the histograms (without equalization first)
show_img_with_matplotlib(image, "image", 1)
show_hist_with_matplotlib_rgb(hist_color, "color histogram", 2, ['b', 'g', 'r'])
show_img_with_matplotlib(added_image, "image lighter", 5)
show_hist_with_matplotlib_rgb(hist_color_added_image, "color histogram", 6, ['b', 'g', 'r'])
show_img_with_matplotlib(subtracted_image, "image darker", 9)
show_hist_with_matplotlib_rgb(hist_color_subtracted_image, "color histogram", 10, ['b', 'g', 'r'])

# Plot the images and the histograms (with equalization)
show_img_with_matplotlib(image_eq, "image equalized", 3)
show_hist_with_matplotlib_rgb(hist_image_eq, "color histogram equalized", 4, ['b', 'g', 'r'])
show_img_with_matplotlib(added_image_eq, "image lighter equalized", 7)
show_hist_with_matplotlib_rgb(hist_added_image_eq, "color histogram equalized", 8, ['b', 'g', 'r'])
show_img_with_matplotlib(subtracted_image_eq, "image darker equalized", 11)
show_hist_with_matplotlib_rgb(hist_subtracted_image_eq, "color histogram equalized", 12, ['b', 'g', 'r'])

# Show the Figure:
plt.show()

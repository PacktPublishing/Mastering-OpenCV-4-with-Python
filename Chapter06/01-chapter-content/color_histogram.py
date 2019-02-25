"""
Color histograms
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


def show_hist_with_matplotlib_rgb(hist, title, pos, color):
    """Shows the histogram using matplotlib capabilities"""

    ax = plt.subplot(2, 3, pos)
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


# Create the dimensions of the figure and set title:
plt.figure(figsize=(15, 6))
plt.suptitle("Color histograms", fontsize=14, fontweight='bold')

# Load the original image:
image = cv2.imread('lenna.png')

# Calculate the histogram for this BGR image:
hist_color = hist_color_img(image)

# Show the BGR image:
show_img_with_matplotlib(image, "image", 1)

# Show the created histogram:
show_hist_with_matplotlib_rgb(hist_color, "color histogram", 4, ['b', 'g', 'r'])

# Add 15 to every pixel on the grayscale image (the result will look lighter) and calculate histogram:
M = np.ones(image.shape, dtype="uint8") * 15
added_image = cv2.add(image, M)
hist_color_added_image = hist_color_img(added_image)

# Subtract 15 from every pixel (the result will look darker) and calculate histogram:
subtracted_image = cv2.subtract(image, M)
hist_color_subtracted_image = hist_color_img(subtracted_image)

# Plot the images and the histograms:
show_img_with_matplotlib(added_image, "image lighter", 2)
show_hist_with_matplotlib_rgb(hist_color_added_image, "color histogram", 5, ['b', 'g', 'r'])
show_img_with_matplotlib(subtracted_image, "image darker", 3)
show_hist_with_matplotlib_rgb(hist_color_subtracted_image, "color histogram", 6, ['b', 'g', 'r'])

# Show the Figure:
plt.show()

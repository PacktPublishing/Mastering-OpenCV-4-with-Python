"""
Bitwise operations (AND, OR) between two loaded images
"""

import argparse
import cv2
import matplotlib.pyplot as plt


def show_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
plt.figure(figsize=(6, 5))
plt.suptitle("Bitwise AND/OR between two images", fontsize=14, fontweight='bold')

# Load the original image (250x250):
image = cv2.imread('lenna_250.png')

# Load the binary image (but as a GBR color image - with 3 channels) (250x250):
binary_image = cv2.imread('opencv_binary_logo_250.png')

# Bitwise AND
bitwise_and = cv2.bitwise_and(image, binary_image)

# Bitwise OR
bitwise_or = cv2.bitwise_or(image, binary_image)

# Display all the resulting images:
show_with_matplotlib(image, "image", 1)
show_with_matplotlib(binary_image, "binary logo", 2)
show_with_matplotlib(bitwise_and, "AND operation", 3)
show_with_matplotlib(bitwise_or, "OR operation", 4)

# Show the Figure:
plt.show()

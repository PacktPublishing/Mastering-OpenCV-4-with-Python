"""
Thresholding example (Otsu's binarization algorithm) using scikit-image
"""

# Import required packages:
import cv2
import matplotlib.pyplot as plt
from skimage.filters import (threshold_otsu, threshold_triangle, threshold_niblack, threshold_sauvola)
from skimage import img_as_ubyte


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 8))
plt.suptitle("Thresholding scikit-image (Otsu, Triangle, Niblack, Sauvola)", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
image = cv2.imread('sudoku.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate histogram (only for visualization):
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Trying Otsu's scikit-image algorithm:
thresh_otsu = threshold_otsu(gray_image)
binary_otsu = gray_image > thresh_otsu
binary_otsu = img_as_ubyte(binary_otsu)

# Trying Niblack's scikit-image algorithm:
thresh_niblack = threshold_niblack(gray_image, window_size=25, k=0.8)
binary_niblack = gray_image > thresh_niblack
binary_niblack = img_as_ubyte(binary_niblack)

# Trying Sauvola's scikit-image algorithm:
thresh_sauvola = threshold_sauvola(gray_image, window_size=25)
binary_sauvola = gray_image > thresh_sauvola
binary_sauvola = img_as_ubyte(binary_sauvola)

# Trying triangle scikit-image algorithm:
thresh_triangle = threshold_triangle(gray_image)
binary_triangle = gray_image > thresh_triangle
binary_triangle = img_as_ubyte(binary_triangle)

# Plot all the images:
show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img", 2)
show_img_with_matplotlib(cv2.cvtColor(binary_otsu, cv2.COLOR_GRAY2BGR), "Otsu's binarization (scikit-image)", 3)
show_img_with_matplotlib(cv2.cvtColor(binary_triangle, cv2.COLOR_GRAY2BGR), "Triangle binarization (scikit-image)", 4)
show_img_with_matplotlib(cv2.cvtColor(binary_niblack, cv2.COLOR_GRAY2BGR), "Niblack's binarization (scikit-image)", 5)
show_img_with_matplotlib(cv2.cvtColor(binary_sauvola, cv2.COLOR_GRAY2BGR), "Sauvola's binarization (scikit-image)", 6)

# Show the Figure:
plt.show()

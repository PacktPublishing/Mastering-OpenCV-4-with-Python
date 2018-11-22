"""
Grayscale histogram comparison
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# Name and path of the test images to load:
image_names = ['gray_image.png', 'gray_blurred.png', 'gray_added_image.png', 'gray_subtracted_image.png']
path = 'comparehist_test_imgs'


# Load all test images building the relative path using 'os.path.join'
def load_all_test_images():
    """Loads all the test images to be used for testing"""

    images = []
    for index_image, name_image in enumerate(image_names):
        # Build the relative path where the current image is:
        image_path = os.path.join(path, name_image)
        # print("image_path: '{}'".format(image_path))
        # Read the image and add it (append) to the structure 'images'
        images.append(cv2.imread(image_path, 0))
    # Return all the loaded test images:
    return images


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(4, 5, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_hist_with_matplotlib_gray(hist, title, pos, color):
    """Shows the histogram using matplotlib capabilities"""

    ax = plt.subplot(2, 5, pos)
    # plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


# Create the dimensions of the figure and set title:
plt.figure(figsize=(18, 9))
plt.suptitle("Grayscale histogram comparison", fontsize=14, fontweight='bold')

# We load all the test images:
test_images = load_all_test_images()
hists = []

# Calculate the histograms for every image:
for img in test_images:
    # Calculate histogram:
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # Normalize histogram:
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    # Add histogram to hists:
    hists.append(hist)

# Perform all the comparisons using all the available metrics, and show the results:
gray_gray = cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_CORREL)
gray_grayblurred = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)
gray_addedgray = cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_CORREL)
gray_subgray = cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_CORREL)

show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "query img", 1)
show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "img 1 " + str('CORREL % 6.5f' % gray_gray),
                         2)
show_img_with_matplotlib(cv2.cvtColor(test_images[1], cv2.COLOR_GRAY2BGR),
                         "img 2 " + str('CORREL % 6.5f' % gray_grayblurred), 3)
show_img_with_matplotlib(cv2.cvtColor(test_images[2], cv2.COLOR_GRAY2BGR),
                         "img 3 " + str('CORREL % 6.5f' % gray_addedgray), 4)
show_img_with_matplotlib(cv2.cvtColor(test_images[3], cv2.COLOR_GRAY2BGR),
                         "img 4 " + str('CORREL % 6.5f' % gray_subgray), 5)

gray_gray = cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_CHISQR)
gray_grayblurred = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CHISQR)
gray_addedgray = cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_CHISQR)
gray_subgray = cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_CHISQR)

show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "query img", 6)
show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "img 1 " + str('CHISQR % 6.5f' % gray_gray),
                         7)
show_img_with_matplotlib(cv2.cvtColor(test_images[1], cv2.COLOR_GRAY2BGR),
                         "img 2 " + str('CHISQR % 6.5f' % gray_grayblurred), 8)
show_img_with_matplotlib(cv2.cvtColor(test_images[2], cv2.COLOR_GRAY2BGR),
                         "img 3 " + str('CHISQR % 6.5f' % gray_addedgray), 9)
show_img_with_matplotlib(cv2.cvtColor(test_images[3], cv2.COLOR_GRAY2BGR),
                         "img 4 " + str('CHISQR % 6.5f' % gray_subgray), 10)

gray_gray = cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_INTERSECT)
gray_grayblurred = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_INTERSECT)
gray_addedgray = cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_INTERSECT)
gray_subgray = cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_INTERSECT)

show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "query img", 11)
show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR),
                         "img 1 " + str('INTERSECT % 6.5f' % gray_gray), 12)
show_img_with_matplotlib(cv2.cvtColor(test_images[1], cv2.COLOR_GRAY2BGR),
                         "img 2 " + str('INTERSECT % 6.5f' % gray_grayblurred), 13)
show_img_with_matplotlib(cv2.cvtColor(test_images[2], cv2.COLOR_GRAY2BGR),
                         "img 3 " + str('INTERSECT % 6.5f' % gray_addedgray), 14)
show_img_with_matplotlib(cv2.cvtColor(test_images[3], cv2.COLOR_GRAY2BGR),
                         "img 4 " + str('INTERSECT % 6.5f' % gray_subgray), 15)

gray_gray = cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_BHATTACHARYYA)
gray_grayblurred = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_BHATTACHARYYA)
gray_addedgray = cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_BHATTACHARYYA)
gray_subgray = cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_BHATTACHARYYA)

show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "query img", 16)
show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR),
                         "img 1 " + str('BHATTACHARYYA % 6.5f' % gray_gray), 17)
show_img_with_matplotlib(cv2.cvtColor(test_images[1], cv2.COLOR_GRAY2BGR),
                         "img 2 " + str('BHATTACHARYYA % 6.5f' % gray_grayblurred), 18)
show_img_with_matplotlib(cv2.cvtColor(test_images[2], cv2.COLOR_GRAY2BGR),
                         "img 3 " + str('BHATTACHARYYA % 6.5f' % gray_addedgray), 19)
show_img_with_matplotlib(cv2.cvtColor(test_images[3], cv2.COLOR_GRAY2BGR),
                         "img 4 " + str('BHATTACHARYYA % 6.5f' % gray_subgray), 20)

# Show the Figure:
plt.show()

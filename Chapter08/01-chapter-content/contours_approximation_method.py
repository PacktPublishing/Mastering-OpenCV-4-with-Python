"""
Approximation methods in OpenCV contours
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def array_to_tuple(arr):
    """Converts array to tuple"""

    return tuple(arr.reshape(1, -1)[0])


def draw_contour_points(img, cnts, color):
    """Draw all points from a list of contours"""

    for cnt in cnts:
        print(cnt.shape)
        squeeze = np.squeeze(cnt)
        print(squeeze.shape)

        for p in squeeze:
            pp = array_to_tuple(p)
            cv2.circle(img, pp, 3, color, -1)

    return img


def build_sample_image():
    """Builds a sample image to search for contours"""

    # Create a 500x500 gray image (70 intensity) with a rectangle and a circle inside:
    img = np.ones((500, 500, 3), dtype="uint8") * 70
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 255), -1)
    cv2.circle(img, (400, 400), 100, (255, 255, 0), -1)

    return img


def build_sample_image_2():
    """Builds a sample image to search for contours"""

    # Create a 500x500 gray image (70 intensity) with a rectangle and a circle inside:
    img = np.ones((500, 500, 3), dtype="uint8") * 70
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 255), -1)
    cv2.rectangle(img, (150, 150), (250, 250), (70, 70, 70), -1)
    cv2.circle(img, (400, 400), 100, (255, 255, 0), -1)
    cv2.circle(img, (400, 400), 50, (70, 70, 70), -1)

    return img


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
plt.suptitle("Contours approximation method", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
image = build_sample_image_2()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply cv2.threshold() to get a ginary image:
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

# Create the different images to plot the detected contours:
image_approx_none = image.copy()
image_approx_simple = image.copy()
image_approx_tc89_l1 = image.copy()
image_approx_tc89_kcos = image.copy()

# Find contours using different methods:
im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
im2, contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
im3, contours3, hierarchy3 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
im4, contours4, hierarchy4 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)

# Draw the contours in the previously created images:
draw_contour_points(image_approx_none, contours, (255, 255, 255))
draw_contour_points(image_approx_simple, contours2, (255, 255, 255))
draw_contour_points(image_approx_tc89_l1, contours3, (255, 255, 255))
draw_contour_points(image_approx_tc89_kcos, contours4, (255, 255, 255))

# Plot all the figures:
show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), "threshold = 100", 2)
show_img_with_matplotlib(image_approx_none, "contours (APPROX_NONE)", 3)
show_img_with_matplotlib(image_approx_simple, "contours (CHAIN_APPROX_SIMPLE)", 4)
show_img_with_matplotlib(image_approx_tc89_l1, "contours (APPROX_TC89_L1)", 5)
show_img_with_matplotlib(image_approx_tc89_kcos, "contours (APPROX_TC89_KCOS)", 6)

# Show the Figure:
plt.show()

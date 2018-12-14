"""
OpenCV functionality related to contours
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def extreme_points_2(contour):
    """Returns extreme points of the contour"""

    extreme_left = tuple(contour[contour[:, :, 0].argmin()][0])
    extreme_right = tuple(contour[contour[:, :, 0].argmax()][0])
    extreme_top = tuple(contour[contour[:, :, 1].argmin()][0])
    extreme_bottom = tuple(contour[contour[:, :, 1].argmax()][0])

    return extreme_left, extreme_right, extreme_top, extreme_bottom


def extreme_points(contour):
    """Returns extreme points of the contour"""

    index_min_x = contour[:, :, 0].argmin()
    index_min_y = contour[:, :, 1].argmin()
    index_max_x = contour[:, :, 0].argmax()
    index_max_y = contour[:, :, 1].argmax()

    extreme_left = tuple(contour[index_min_x][0])
    extreme_right = tuple(contour[index_max_x][0])
    extreme_top = tuple(contour[index_min_y][0])
    extreme_bottom = tuple(contour[index_max_y][0])

    return extreme_left, extreme_right, extreme_top, extreme_bottom


def array_to_tuple(arr):
    """Converts array to tuple"""

    return tuple(arr.reshape(1, -1)[0])


def draw_contour_points(img, cnts, color):
    """Draw all points from a list of contours"""

    for cnt in cnts:
        # print(cnt.shape)
        squeeze = np.squeeze(cnt)
        # print(squeeze.shape)

        for p in squeeze:
            pp = array_to_tuple(p)
            cv2.circle(img, pp, 10, color, -1)

    return img


def draw_contour_outline(img, cnts, color, thickness=1):
    """ Draws contours outlines of each contour """

    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(13, 9))
plt.suptitle("Functionality related to contours", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
image = cv2.imread("shape_features.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply cv2.threshold() to get a binary image:
ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

# Find contours using the thresholded image:
im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Show the number of detected contours:
print("detected contours: '{}' ".format(len(contours)))

# Create copy of the original image to perform the visualization of each operation:
boundingRect_image = image.copy()
minAreaRect_image = image.copy()
fitEllipse_image = image.copy()
minEnclosingCircle_image = image.copy()
approxPolyDP_image = image.copy()

# 1. cv2.boundingRect():
x, y, w, h = cv2.boundingRect(contours[0])
cv2.rectangle(boundingRect_image, (x, y), (x + w, y + h), (0, 255, 0), 5)

# 2. cv2.minAreaRect():
rotated_rect = cv2.minAreaRect(contours[0])
box = cv2.boxPoints(rotated_rect)
box = np.int0(box)
cv2.polylines(minAreaRect_image, [box], True, (0, 0, 255), 5)

# 3. cv2.minEnclosingCircle():
(x, y), radius = cv2.minEnclosingCircle(contours[0])
center = (int(x), int(y))
radius = int(radius)
cv2.circle(minEnclosingCircle_image, center, radius, (255, 0, 0), 5)

# 4. cv2.fitEllipse():
ellipse = cv2.fitEllipse(contours[0])
cv2.ellipse(fitEllipse_image, ellipse, (0, 255, 255), 5)

# 5. cv2.approxPolyDP():
epsilon = 0.01 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[0], epsilon, True)
draw_contour_outline(approxPolyDP_image, [approx], (255, 255, 0), 5)
draw_contour_points(approxPolyDP_image, [approx], (255, 0, 255))

# 6. Detect extreme points of the contour:
left, right, top, bottom = extreme_points_2(contours[0])
cv2.circle(image, left, 20, (255, 0, 0), -1)
cv2.circle(image, right, 20, (0, 255, 0), -1)
cv2.circle(image, top, 20, (0, 255, 255), -1)
cv2.circle(image, bottom, 20, (0, 0, 255), -1)

# Plot the image:
show_img_with_matplotlib(image, "image and extreme points", 1)
show_img_with_matplotlib(boundingRect_image, "cv2.boundingRect()", 2)
show_img_with_matplotlib(minAreaRect_image, "cv2.minAreaRect()", 3)
show_img_with_matplotlib(minEnclosingCircle_image, "cv2.minEnclosingCircle()", 4)
show_img_with_matplotlib(fitEllipse_image, "cv2.ellipse()", 5)
show_img_with_matplotlib(approxPolyDP_image, "cv2.approxPolyDP()", 6)

# Show the Figure:
plt.show()

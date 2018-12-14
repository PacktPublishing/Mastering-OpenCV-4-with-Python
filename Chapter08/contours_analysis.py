"""
Contours analysis based mainly on moments
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


# the aspect ratio is defined as: Width/Height
def aspect_ratio(contour):
    """Returns the aspect ratio of the contour based on the dimensions of the bounding rect"""

    x, y, w, h = cv2.boundingRect(contour)
    res = float(w) / h
    return res


def roundness(contour, moments):
    """Calculates the roundness of a contour"""

    length = cv2.arcLength(contour, True)
    k = (length * length) / (moments['m00'] * 4 * np.pi)
    return k


def eccentricity_from_ellipse(contour):
    """Calculates the eccentricity fitting an ellipse from a contour"""

    (x, y), (MA, ma), angle = cv2.fitEllipse(contour)

    a = ma / 2
    b = MA / 2

    ecc = np.sqrt(a ** 2 - b ** 2) / a
    return ecc


def eccentricity_from_moments(moments):
    """Calculates the eccentricity from the moments of the contour"""

    a1 = (moments['mu20'] + moments['mu02']) / 2
    a2 = np.sqrt(4 * moments['mu11'] ** 2 + (moments['mu20'] - moments['mu02']) ** 2) / 2
    ecc = np.sqrt(1 - (a1 - a2) / (a1 + a2))
    return ecc


def get_one_contour():
    """Returns a 'fixed' contour"""

    cnts = [np.array(
        [[[600, 320]], [[563, 460]], [[460, 562]], [[320, 600]], [[180, 563]], [[78, 460]], [[40, 320]], [[77, 180]],
         [[179, 78]], [[319, 40]], [[459, 77]], [[562, 179]]], dtype=np.int32)]
    return cnts


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
    """Draws contours outlines of each contour"""

    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 5))
plt.suptitle("Contour analysis", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Create the canvas (black image with three channels):
canvas = np.zeros((640, 640, 3), dtype="uint8")

# Get a sample contours:
contours = get_one_contour()
print("'detected' contours: '{}' ".format(len(contours)))

# Create copy images to show the different results:
image_contour_points = canvas.copy()
image_contour_outline = canvas.copy()
image_contour_points_outline = canvas.copy()

# Draw only contour points:
draw_contour_points(image_contour_points, contours, (255, 0, 255))

# Draw only contour outline:
draw_contour_outline(image_contour_outline, contours, (0, 255, 255), -1)

# Draw both contour outline and points:
draw_contour_outline(image_contour_points_outline, contours, (255, 0, 0), 3)
draw_contour_points(image_contour_points_outline, contours, (0, 0, 255))

# Calculate image moments of the detected contour
M = cv2.moments(contours[0])
print("moments calculated from the detected contour: {}".format(M))

# 1) Calculate/show contour area using both cv2.contourArea() or m00 moment:
print("Contour area: '{}'".format(cv2.contourArea(contours[0])))
print("Contour area: '{}'".format(M['m00']))

# 2) Calculate centroid:
x_centroid = round(M['m10'] / M['m00'])
y_centroid = round(M['m01'] / M['m00'])
print("center X : '{}'".format(x_centroid))
print("center Y : '{}'".format(y_centroid))

# Draw circle based on the centroid coordinates:
cv2.circle(image_contour_points, (x_centroid, y_centroid), 10, (255, 255, 255), -1)

# 3) Calculate roundness (k):
# roundness (k) = (perimeter * perimeter) / (Area * 4 * PI):
# Therefore k for a circle is equal 1, for other objects > 1.
k = roundness(contours[0], M)
print("roundness: '{}'".format(k))

# 4) Calculate eccentricity using both provided formulas:
em = eccentricity_from_moments(M)
print("eccentricity: '{}'".format(em))
ee = eccentricity_from_ellipse(contours[0])
print("eccentricity: '{}'".format(ee))

# 5) Calculate the aspect ratio of the contour:
ar = aspect_ratio(contours[0])
print("aspect ratio: '{}'".format(ar))

# Plot the images:
show_img_with_matplotlib(image_contour_points, "centroid : (" + str(x_centroid) + "," + str(y_centroid) + ")", 1)
show_img_with_matplotlib(image_contour_outline, "size: " + str(M['m00']) + " & aspect ratio: " + str(ar), 2)
show_img_with_matplotlib(image_contour_points_outline,
                         "roundness: " + str(round(k, 3)) + " & eccentricity: " + str(round(ee, 3)), 3)

# Show the Figure:
plt.show()

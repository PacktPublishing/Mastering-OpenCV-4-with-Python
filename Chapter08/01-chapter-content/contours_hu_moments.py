"""
Hu moments calculation
"""

import cv2
from matplotlib import pyplot as plt


def centroid(moments):
    """Returns centroid based on momments"""

    x_centroid = round(moments['m10'] / moments['m00'])
    y_centroid = round(moments['m01'] / moments['m00'])
    return x_centroid, y_centroid


def draw_contour_outline(img, cnts, color, thickness=1):
    """Draws contours outlines of each contour"""

    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 5))
plt.suptitle("Hu moments", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
image = cv2.imread("shape_features.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply cv2.threshold() to get a binary image:
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

# Compute moments:
M = cv2.moments(thresh, True)
print("moments: '{}'".format(M))

# Calculate the centroid of the contour based on moments:
x, y = centroid(M)

# Compute Hu moments:
HuM = cv2.HuMoments(M)
print("Hu moments: '{}'".format(HuM))

# Find contours in the thresholded image:
im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Compute moments:
M2 = cv2.moments(contours[0])
print("moments: '{}'".format(M2))

# Calculate the centroid of the contour based on moments:
x2, y2 = centroid(M2)

# Compute Hu moments:
HuM2 = cv2.HuMoments(M2)
print("Hu moments: '{}'".format(HuM2))

# Draw the outline of the detected contour:
draw_contour_outline(image, contours, (255, 0, 0), 10)

# Draw the centroids (it should be the same point):
# (make it big to see the difference)
cv2.circle(image, (x, y), 25, (255, 0, 0), -1)
cv2.circle(image, (x2, y2), 25, (0, 255, 0), -1)
print("('x','y'): ('{}','{}')".format(x, y))
print("('x2','y2'): ('{}','{}')".format(x2, y2))

# Plot the images:
show_img_with_matplotlib(image, "detected contour and centroid", 1)

# Show the Figure:
plt.show()

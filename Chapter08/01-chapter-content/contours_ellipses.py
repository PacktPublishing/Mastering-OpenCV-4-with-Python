"""
Some features based on moments are calculated
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def roundness(contour, moments):
    """Calculates the roundness of a contour"""

    length = cv2.arcLength(contour, True)
    k = (length * length) / (moments['m00'] * 4 * np.pi)
    return k


def get_position_to_draw(text, point, font_face, font_scale, thickness):
    """Gives the coordinates to draw centered"""

    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)


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


def build_image_ellipses():
    """Draws ellipses in the image"""

    img = np.zeros((500, 600, 3), dtype="uint8")
    cv2.ellipse(img, (120, 60), (100, 50), 0, 0, 360, (255, 255, 0), -1)
    cv2.ellipse(img, (300, 60), (50, 50), 0, 0, 360, (0, 0, 255), -1)
    cv2.ellipse(img, (425, 200), (50, 150), 0, 0, 360, (255, 0, 0), -1)
    cv2.ellipse(img, (550, 250), (20, 240), 0, 0, 360, (255, 0, 255), -1)
    cv2.ellipse(img, (200, 200), (150, 50), 0, 0, 360, (0, 255, 0), -1)
    cv2.ellipse(img, (250, 400), (200, 50), 0, 0, 360, (0, 255, 255), -1)
    return img


def draw_contour_outline(img, cnts, color, thickness=1):
    """Draws contours outlines of each contour"""

    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(14, 6))
plt.suptitle("Eccentricity", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
# image = build_sample_image_2()
image = build_image_ellipses()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply cv2.threshold() to get a binary image:
ret, thresh = cv2.threshold(gray_image, 20, 255, cv2.THRESH_BINARY)

# Find contours using the thresholded image:
im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Show the number of detected contours:
print("detected contours: '{}' ".format(len(contours)))

# Create a copy to show the results:
img_numbers = image.copy()

for contour in contours:
    # Draw the contour:
    draw_contour_outline(image, [contour], (255, 255, 255), 5)

    # Compute the moments of the contour:
    M = cv2.moments(contour)

    # Calculate the roundness:
    k = roundness(contour, M)
    print("roundness: '{}'".format(k))

    # Calculate eccentricy using the two provided formulas:
    em = eccentricity_from_moments(M)
    print("eccentricity: '{}'".format(em))
    ee = eccentricity_from_ellipse(contour)
    print("eccentricity: '{}'".format(ee))

    # Get centroid of the contour:
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    # Ge get the text to draw:
    text_to_draw = str(round(em, 3))

    # Get the position to draw:
    (x, y) = get_position_to_draw(text_to_draw, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, 3)

    # Write the name of shape on the center of shapes:
    cv2.putText(img_numbers, text_to_draw, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

# Plot the images:
show_img_with_matplotlib(image, "image", 1)
show_img_with_matplotlib(img_numbers, "ellipses eccentricity", 2)

# Show the Figure:
plt.show()

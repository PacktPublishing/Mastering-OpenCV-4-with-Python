"""
Matching contours using cv2.matchShapes() against a perfect circle
"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt


def get_position_to_draw(text, point, font_face, font_scale, thickness):
    """Gives the coordinates to draw centered"""

    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)


def build_circle_image():
    """Builds a circle image"""

    # Create a 500x500 black image with a circle inside:
    img = np.zeros((500, 500, 3), dtype="uint8")
    cv2.circle(img, (250, 250), 200, (255, 255, 255), 1)

    return img


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(18, 6))
plt.suptitle("Matching contours (against a perfect circle) using cv2.matchShapes()", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the images and convert them to grayscale:
image = cv2.imread("match_shapes.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_circle = build_circle_image()
gray_image_circle = cv2.cvtColor(image_circle, cv2.COLOR_BGR2GRAY)

# Apply cv2.threshold() to get binary images:
ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY_INV)
ret, thresh_circle = cv2.threshold(gray_image_circle, 70, 255, cv2.THRESH_BINARY)

# Find contours using the thresholded images:
im, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
im_2, contours_circle, hierarchy_2 = cv2.findContours(thresh_circle, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Copy three images to show the results:
result_1 = image.copy()
result_2 = image.copy()
result_3 = image.copy()

# At this point we compare all the detected contours with the circle contour to get the similarity of the match
for contour in contours:
    # Compute the moment of contour:
    M = cv2.moments(contour)

    # The center or centroid can be calculated as follows:
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    # We match each contour against the circle contour using the three matching modes:
    ret_1 = cv2.matchShapes(contours_circle[0], contour, cv2.CONTOURS_MATCH_I1, 0.0)
    ret_2 = cv2.matchShapes(contours_circle[0], contour, cv2.CONTOURS_MATCH_I2, 0.0)
    ret_3 = cv2.matchShapes(contours_circle[0], contour, cv2.CONTOURS_MATCH_I3, 0.0)

    # Get the positions to draw:
    (x_1, y_1) = get_position_to_draw(str(round(ret_1, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    (x_2, y_2) = get_position_to_draw(str(round(ret_2, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    (x_3, y_3) = get_position_to_draw(str(round(ret_3, 3)), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)

    # Write the obtainted scores in the result images:
    cv2.putText(result_1, str(round(ret_1, 3)), (x_1, y_1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
    cv2.putText(result_2, str(round(ret_2, 3)), (x_2, y_2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(result_3, str(round(ret_3, 3)), (x_3, y_3), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

# Plot the images:
show_img_with_matplotlib(result_1, "matching scores (method = CONTOURS_MATCH_I1)", 1)
show_img_with_matplotlib(result_2, "matching scores (method = CONTOURS_MATCH_I2)", 2)
show_img_with_matplotlib(result_3, "matching scores (method = CONTOURS_MATCH_I3)", 3)

# Show the Figure:
plt.show()

"""
Ordering contours based on the size
"""

import cv2
from matplotlib import pyplot as plt


def get_position_to_draw(text, point, font_face, font_scale, thickness):
    """Gives the coordinates to draw centered"""

    text_size = cv2.getTextSize(text, font_face, font_scale, thickness)[0]
    text_x = point[0] - text_size[0] / 2
    text_y = point[1] + text_size[1] / 2
    return round(text_x), round(text_y)


def sort_contours_size(cnts):
    """Sort contours based on the size"""

    cnts_sizes = [cv2.contourArea(contour) for contour in cnts]
    (cnts_sizes, cnts) = zip(*sorted(zip(cnts_sizes, cnts)))
    return cnts_sizes, cnts


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(9, 9))
plt.suptitle("Sort contours by size", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale:
# image = build_sample_image_2()
image = cv2.imread("shapes_sizes.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Plot the image:
show_img_with_matplotlib(image, "image", 1)

# Apply cv2.threshold() to get a binary image
ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

# Find contours using the thresholded image:
# Note: cv2.findContours() has been changed to return only the contours and the hierarchy
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Show the number of detected contours:
print("detected contours: '{}' ".format(len(contours)))

# Sort the contours based on the size:
(contour_sizes, contours) = sort_contours_size(contours)

for i, (size, contour) in enumerate(zip(contour_sizes, contours)):
    # Compute the moment of contour:
    M = cv2.moments(contour)

    # The center or centroid can be calculated as follows:
    cX = int(M['m10'] / M['m00'])
    cY = int(M['m01'] / M['m00'])

    # Get the position to draw:
    (x, y) = get_position_to_draw(str(i + 1), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 2, 5)

    # Write the ordering of the shape on the center of shapes
    cv2.putText(image, str(i + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)

# Plot the image
show_img_with_matplotlib(image, "result", 2)

# Show the Figure:
plt.show()

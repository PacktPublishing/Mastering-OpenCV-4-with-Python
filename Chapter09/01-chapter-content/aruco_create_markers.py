"""
This script generates some marker images ready to be printed using Aruco
"""

# Import required packages
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """ Shows an image using matplotlib capabilities """

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 5))
plt.suptitle("Aruco markers creation", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# The first step is to create the dictionary object. Aruco has some predefined dictionaries.
# (DICT_4X4_100, DICT_4X4_1000, DICT_4X4_250, DICT_4X4_50 = 0, .... , DICT_7X7_1000)
# We are going to create a dictionary, which is composed by 250 markers.
# Each marker will be of 7x7 bits (DICT_7X7_250):
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

# Now we can draw a marker using 'cv2.aruco.drawMarker()'.
# The function 'cv2.aruco.drawMarker()' returns the marker image ready to be printed (in a canonical form)
# The first parameter is the dictionary object 'aruco_dictionary' we have created previously
# using 'cv2.aruco.Dictionary_get()'
# The second parameter is the marker id, which ranges between 0 and 249 (our dictionary has 250 markers)
# The third parameter is the size of the image to be drawn. in this case, the marker will have a size of 600x600 pixels
# The fourth (optional, by default 1) parameter is the number of bits in marker borders
aruco_marker_1 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=1)
aruco_marker_2 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=2)
aruco_marker_3 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=3)

# We save the created markers using 'cv2.imwrite()':
cv2.imwrite("marker_DICT_7X7_250_600_1.png", aruco_marker_1)
cv2.imwrite("marker_DICT_7X7_250_600_2.png", aruco_marker_2)
cv2.imwrite("marker_DICT_7X7_250_600_3.png", aruco_marker_3)

# Plot the images:
show_img_with_matplotlib(cv2.cvtColor(aruco_marker_1, cv2.COLOR_GRAY2BGR), "marker_DICT_7X7_250_600_1", 1)
show_img_with_matplotlib(cv2.cvtColor(aruco_marker_2, cv2.COLOR_GRAY2BGR), "marker_DICT_7X7_250_600_2", 2)
show_img_with_matplotlib(cv2.cvtColor(aruco_marker_3, cv2.COLOR_GRAY2BGR), "marker_DICT_7X7_250_600_3", 3)

# Show the Figure:
plt.show()

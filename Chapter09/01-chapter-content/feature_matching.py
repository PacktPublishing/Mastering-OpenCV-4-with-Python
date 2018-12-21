"""
Feature matching using ORB descriptors and Brute-Force (BF) matcher
"""

# Import required packages:
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """ Shows an image using matplotlib capabilities """

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(8, 6))
plt.suptitle("ORB descriptors and Brute-Force (BF) matcher", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the 'query' and 'scene' image:
image_query = cv2.imread('opencv_logo_with_text.png')
image_scene = cv2.imread('opencv_logo_with_text_scene.png')

# Initiate ORB detector:
orb = cv2.ORB_create()

# Detect the keypoints and compute the descriptors with ORB:
keypoints_1, descriptors_1 = orb.detectAndCompute(image_query, None)
keypoints_2, descriptors_2 = orb.detectAndCompute(image_scene, None)

# Create BFMatcher object
# First parameter sets the distance measurement (by default it is cv2.NORM_L2)
# The second parameter crossCheck (which is False by default) can be set to True in order to return only
# consistent pairs in the matching process (the two features in both sets should match each other)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors:
bf_matches = bf_matcher.match(descriptors_1, descriptors_2)

# Sort the matches in the order of their distance:
bf_matches = sorted(bf_matches, key=lambda x: x.distance)

# Draw first 20 matches:
result = cv2.drawMatches(image_query, keypoints_1, image_scene, keypoints_2, bf_matches[:20], None,
                         matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)

# Plot the images:
show_img_with_matplotlib(result, "matches between the two images", 1)

# Show the Figure:
plt.show()

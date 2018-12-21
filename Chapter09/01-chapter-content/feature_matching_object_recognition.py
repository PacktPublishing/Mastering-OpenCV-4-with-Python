"""
Feature detection, matching and object recognition based on ORB descriptors and BF matcher
"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """ Shows an image using matplotlib capabilities """

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Load the 'query' and 'scene' image:
image_query = cv2.imread('opencv_logo_with_text.png')
image_scene = cv2.imread('opencv_logo_with_text_scene.png')

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(14, 5))
plt.suptitle("Feature matching and homography computation for object recognition", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

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
best_matches = bf_matches[:40]

# Extract the matched keypoints:
pts_src = np.float32([keypoints_1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
pts_dst = np.float32([keypoints_2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# Find homography matrix:
M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)

# Get the corner coordinates of the 'query' image:
h, w = image_query.shape[:2]
pts_corners_src = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

# Perform perspective transform using the previously calculated matrix and the corners of the 'query' image
# to get the corners of the 'detected' object in the 'scene' image:
pts_corners_dst = cv2.perspectiveTransform(pts_corners_src, M)

# Draw corners of the detected object:
img_obj = cv2.polylines(image_scene, [np.int32(pts_corners_dst)], True, (0, 255, 255), 10)

# Draw matches:
img_matching = cv2.drawMatches(image_query, keypoints_1, img_obj, keypoints_2, best_matches, None,
                               matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)

# Plot the images:
show_img_with_matplotlib(img_obj, "detected object", 1)
show_img_with_matplotlib(img_matching, "feature matching", 2)

# Show the Figure:
plt.show()

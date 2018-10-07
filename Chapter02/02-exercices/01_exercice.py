"""
How to check if img is color or grayscale
"""

# import required packages
import cv2


# load OpenCV logo image:
img = cv2.imread('logo.png')

# Get the shape of the image:
dimensions = img.shape

# Color images: length == 3
# Grayscale images: length == 2
# Check the length of dimensions
if len(dimensions) < 3:
    print("grayscale image!")
if len(dimensions) == 3:
    print("color image!")

# Load the same image but in grayscale:
gray_img = cv2.imread('logo.png', cv2.IMREAD_GRAYSCALE)

# Get again the img.shape properties:
dimensions = gray_img.shape

# Check the length of dimensions
if len(dimensions) < 3:
    print("grayscale image!")
if len(dimensions) == 3:
    print("color image!")

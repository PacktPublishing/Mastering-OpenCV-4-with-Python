"""
Example for testing all colors maps in OpenCV
"""

# Import required packages:
import cv2
import matplotlib.pyplot as plt


def show_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 7, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# We load the image using cv2.imread() and using 'cv2.IMREAD_GRAYSCALE' argument:
gray_img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

# We define all the color map names to be used later:
colormaps = ["AUTUMN", "BONE", "JET", "WINTER", "RAINBOW", "OCEAN", "SUMMER", "SPRING", "COOL", "HSV", "HOT", "PINK",
             "PARULA"]

# We create a figure() object with appropriate size and title
plt.figure(figsize=(12, 5))
plt.suptitle("Colormaps", fontsize=14, fontweight='bold')

# First, we add the grayscale image to the Figure:
# Note that we convert ir to BGR for simplicity and make use of the function 'show_with_matplotlib'
# The function 'show_with_matplotlib' receives a BGR image:
show_with_matplotlib(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR), "GRAY", 1)

# Now we iterate to apply all the colormaps and add the Figure:
for idx, val in enumerate(colormaps):
    # print("idx: {}, val: {}".format(idx, val))
    show_with_matplotlib(cv2.applyColorMap(gray_img, idx), val, idx + 2)

# Show the created image:
plt.show()

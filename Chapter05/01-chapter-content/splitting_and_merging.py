"""
Splitting and merging channels
"""

# Import required packages
import cv2
import matplotlib.pyplot as plt


def show_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 6, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Load the original image:
image = cv2.imread('color_spaces.png')

# create a figure() object with appropriate size and title:
plt.figure(figsize=(13, 5))
plt.suptitle("Splitting and merging channels in OpenCV", fontsize=14, fontweight='bold')

# Show the BGR image:
show_with_matplotlib(image, "BGR - image", 1)

# Split the image into its three components (blue, green and red):
(b, g, r) = cv2.split(image)

# Show all the channels from the BGR image:
show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR - (B)", 2)
show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR - (G)", 2 + 6)
show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR - (R)", 2 + 6 * 2)

# Merge the three channels again to build a BGR image:
image_copy = cv2.merge((b, g, r))

# Show the BGR image:
show_with_matplotlib(image_copy, "BGR - image (copy)", 1 + 6)

# You should remember that cv2.split() is a time consuming operation
# Therefore, you should only use it if it is strictly necessary
# Otherwise, you can use numpy functionality to work with specific channels
# Another way of getting one component (in this case, the blue one)
# is using numpy idexing:
b_copy = image[:, :, 0]

# We make a copy of the loaded image:
image_without_blue = image.copy()

# From the BGR image, we "eliminate" (set to 0) the blue component (channel 0):
image_without_blue[:, :, 0] = 0

# From the BGR image, we "eliminate" (set to 0) the green component (channel 1):
image_without_green = image.copy()
image_without_green[:, :, 1] = 0

# From the BGR image, we "eliminate" (set to 0) the red component (channel 2):
image_without_red = image.copy()
image_without_red[:, :, 2] = 0

# Show all the channels from the BGR image:
show_with_matplotlib(image_without_blue, "BGR without B", 3)
show_with_matplotlib(image_without_green, "BGR without G", 3 + 6)
show_with_matplotlib(image_without_red, "BGR without R", 3 + 6 * 2)

# Split the 'image_without_blue' image into its three components (blue, green and red):
(b, g, r) = cv2.split(image_without_blue)

# Show all the channels from the BGR image without the blue information:
show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR without B (B)", 4)
show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR without B (G)", 4 + 6)
show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR without B (R)", 4 + 6 * 2)

# Split the 'image_without_green' image into its three components (blue, green and red):
(b, g, r) = cv2.split(image_without_green)

# Show all the channels from the BGR image without the green information:
show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR without G (B)", 5)
show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR without G (G)", 5 + 6)
show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR without G (R)", 5 + 6 * 2)

# Split the 'image_without_red' image into its three components (blue, green and red):
(b, g, r) = cv2.split(image_without_red)

# Show all the channels from the BGR image without the red information:
show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR without R (B)", 6)
show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR without R (G)", 6 + 6)
show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR without R (R)", 6 + 6 * 2)

# Show the created image:
plt.show()

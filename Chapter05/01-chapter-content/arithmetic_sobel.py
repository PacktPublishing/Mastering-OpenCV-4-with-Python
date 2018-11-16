"""
Sobel operator example in order to see how this operator works and how cv2.addWeighted() can be used
"""

# Import required packages:
import cv2
import matplotlib.pyplot as plt


def show_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create the dimensions of the figure and set title:
plt.figure(figsize=(10, 4))
plt.suptitle("Sobel operator and cv2.addWeighted() to show the output", fontsize=14, fontweight='bold')

# Load the original image:
image = cv2.imread('lenna.png')

# Filter the image with a gaussian kernel:
image_filtered = cv2.GaussianBlur(image, (3, 3), 0)

# We convert the image to grayscale:
gray_image = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2GRAY)

# Gradient x is calculated:
# the depth of the output is set to CV_16S to avoid overflow
# CV_16S = one channel of 2-byte signed integers (16-bit signed integers)
gradient_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0, 3)

# Gradient y is calculated:
# the depth of the output is set to CV_16S to avoid overflow
# CV_16S = one channel of 2-byte signed integers (16-bit signed integers)
gradient_y = cv2.Sobel(gray_image, cv2.CV_16S, 0, 1, 3)

# Conversion to an unsigned 8-bit type:
abs_gradient_x = cv2.convertScaleAbs(gradient_x)
abs_gradient_y = cv2.convertScaleAbs(gradient_y)

# Combine the two images using the same weight:
sobel_image = cv2.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)

# Display all the resulting images:
show_with_matplotlib(image, "Image", 1)
show_with_matplotlib(cv2.cvtColor(abs_gradient_x, cv2.COLOR_GRAY2BGR), "Gradient x", 2)
show_with_matplotlib(cv2.cvtColor(abs_gradient_y, cv2.COLOR_GRAY2BGR), "Gradient y", 3)
show_with_matplotlib(cv2.cvtColor(sobel_image, cv2.COLOR_GRAY2BGR), "Sobel output", 4)

# Show the Figure:
plt.show()

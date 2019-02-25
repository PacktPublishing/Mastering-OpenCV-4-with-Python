"""
Comparing different methods for sharpening images
"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def unsharped_filter(img):
    """The unsharp filter enhances edges subtracting the smoothed image from the original image"""

    smoothed = cv2.GaussianBlur(img, (9, 9), 10)
    return cv2.addWeighted(img, 1.5, smoothed, -0.5, 0)


# Create the dimensions of the figure and set title:
plt.figure(figsize=(12, 6))
plt.suptitle("Sharpening images", fontsize=14, fontweight='bold')

# Load the image:
image = cv2.imread('cat-face.png')

# We create the kernel for sharpening images
kernel_sharpen_1 = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])

kernel_sharpen_2 = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])

kernel_sharpen_3 = np.array([[1, 1, 1],
                             [1, -7, 1],
                             [1, 1, 1]])

kernel_sharpen_4 = np.array([[-1, -1, -1, -1, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, 2, 8, 2, -1],
                             [-1, 2, 2, 2, -1],
                             [-1, -1, -1, -1, -1]]) / 8.0

# Apply all the kernels we have created:
sharp_image_1 = cv2.filter2D(image, -1, kernel_sharpen_1)
sharp_image_2 = cv2.filter2D(image, -1, kernel_sharpen_2)
sharp_image_3 = cv2.filter2D(image, -1, kernel_sharpen_3)
sharp_image_4 = cv2.filter2D(image, -1, kernel_sharpen_4)

# Try the unsharped filter:
sharp_image_5 = unsharped_filter(image)

# Display all the resulting images:
show_with_matplotlib(image, "original", 1)
show_with_matplotlib(sharp_image_1, "sharp 1", 2)
show_with_matplotlib(sharp_image_2, "sharp 2", 3)
show_with_matplotlib(sharp_image_3, "sharp 3", 4)
show_with_matplotlib(sharp_image_4, "sharp 4", 5)
show_with_matplotlib(sharp_image_5, "sharp 5", 6)

# Show the Figure:
plt.show()

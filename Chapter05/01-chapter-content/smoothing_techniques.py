"""
Comparing different methods for smoothing images
"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Create a figure() object with appropriate size and title
plt.figure(figsize=(12, 6))
plt.suptitle("Smoothing techniques", fontsize=14, fontweight='bold')

image = cv2.imread('cat-face.png')

# We create the kernel for smoothing images
# In this case a (10,10) kernel is created
kernel_averaging_10_10 = np.ones((10, 10), np.float32) / 100

# Additionally, if you know the values, you can put them directly in the kernel:
# kernel_averaging_5_5 = np.ones((5, 5), np.float32)/25
kernel_averaging_5_5 = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                                 [0.04, 0.04, 0.04, 0.04, 0.04],
                                 [0.04, 0.04, 0.04, 0.04, 0.04],
                                 [0.04, 0.04, 0.04, 0.04, 0.04],
                                 [0.04, 0.04, 0.04, 0.04, 0.04]])

print("kernel: {}".format(kernel_averaging_5_5))

# The function cv2.filter2D() applies an arbitrary linear filter to the provided image:
smooth_image_f2D_5_5 = cv2.filter2D(image, -1, kernel_averaging_5_5)
smooth_image_f2D_10_10 = cv2.filter2D(image, -1, kernel_averaging_10_10)

# The function cv2.blur() smooths an image using the normalized box filter
smooth_image_b = cv2.blur(image, (10, 10))

# When the parameter normalize (by default True) of cv2.boxFilter() is equals to True,
# cv2.filter2D() and cv2.boxFilter() perform the same operation:
smooth_image_bfi = cv2.boxFilter(image, -1, (10, 10), normalize=True)

# The function cv2.GaussianBlur() convolves the source image with the specified Gaussian kernel
# This kernel can be controlled using the parameters ksize (kernel size),
# sigmaX(standard deviation in the x direction of the gaussian kernel) and
# sigmaY (standard deviation in the y direction of the gaussian kernel)
smooth_image_gb = cv2.GaussianBlur(image, (9, 9), 0)

# The function cv2.medianBlur(), which blurs the image with a median kernel:
smooth_image_mb = cv2.medianBlur(image, 9)

# The function cv2.bilateralFilter() can be applied to the input image in order to apply a bilateral filter:
smooth_image_bf = cv2.bilateralFilter(image, 5, 10, 10)
smooth_image_bf_2 = cv2.bilateralFilter(image, 9, 200, 200)

# Plot the images:
show_with_matplotlib(image, "original", 1)
show_with_matplotlib(smooth_image_f2D_5_5, "cv2.filter2D() (5,5) kernel", 2)
show_with_matplotlib(smooth_image_f2D_10_10, "cv2.filter2D() (10,10) kernel", 3)
show_with_matplotlib(smooth_image_b, "cv2.blur()", 4)
show_with_matplotlib(smooth_image_bfi, "cv2.boxFilter()", 5)
show_with_matplotlib(smooth_image_gb, "cv2.GaussianBlur()", 6)
show_with_matplotlib(smooth_image_mb, "cv2.medianBlur()", 7)
show_with_matplotlib(smooth_image_bf, "cv2.bilateralFilter() - small values", 8)
show_with_matplotlib(smooth_image_bf_2, "cv2.bilateralFilter() - big values", 9)

# Show the created image:
plt.show()

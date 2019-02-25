"""
Comparing how to create histograms using OpenCV, numpy and matplotlib
"""

# Import required packages:
import numpy as np
import cv2
from matplotlib import pyplot as plt
from timeit import default_timer as timer


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_hist_with_matplotlib_gray(hist, title, pos, color):
    """Shows the histogram using matplotlib capabilities"""

    ax = plt.subplot(1, 4, pos)
    plt.title(title)
    plt.xlabel("bins")
    plt.ylabel("number of pixels")
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


# Create the dimensions of the figure and set title:
plt.figure(figsize=(18, 6))
plt.suptitle("Comparing histogram (OpenCV, numpy, matplotlib)", fontsize=14, fontweight='bold')

# Load the image and convert it to grayscale:
image = cv2.imread('lenna.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Take the execution time (execution_time = end - start) for cv2.calcHist():
start = timer()
# Calculate the histogram calling cv2.calcHist()
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
end = timer()
exec_time_calc_hist = (end - start) * 1000

# Take the execution time (execution_time = end - start) for np.histogram():
start = timer()
# Calculate the histogram calling np.histogram():
hist_np, bins_np = np.histogram(gray_image.ravel(), 256, [0, 256])
end = timer()
exec_time_np_hist = (end - start) * 1000

# Take the execution time (execution_time = end - start) for plt.hist():
start = timer()
# Calculate the histogram calling plt.hist():
(n, bins, patches) = plt.hist(gray_image.ravel(), 256, [0, 256])
end = timer()
exec_time_plt_hist = (end - start) * 1000

# Plot the grayscale image and the histogram:
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_hist_with_matplotlib_gray(hist, "grayscale histogram (OpenCV)-" + str('% 6.2f ms' % exec_time_calc_hist), 2, 'm')
show_hist_with_matplotlib_gray(hist_np, "grayscale histogram (Numpy)-" + str('% 6.2f ms' % exec_time_np_hist), 3, 'm')
show_hist_with_matplotlib_gray(n, "grayscale histogram (Matplotlib)-" + str('% 6.2f ms' % exec_time_plt_hist), 4, 'm')

# Show the Figure:
plt.show()

"""
Example to show the lineType argument in OpenCV
"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_with_matplotlib(img, title):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB:
    img_RGB = img[:, :, ::-1]

    # Show the image using matplotlib:
    plt.imshow(img_RGB)
    plt.title(title)
    plt.show()


# Dictionary containing some colors:
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

# We create the canvas to draw: 20 x 20 pixels, 3 channels, uint8 (8-bit unsigned integers)
# We set the background to black using np.zeros():
image = np.zeros((20, 20, 3), dtype="uint8")

# If you want another background color you can do the following:
image[:] = colors['light_gray']

# We are going to see how cv2.line() works modifying the parameter lineType:
cv2.line(image, (5, 0), (20, 15), colors['yellow'], 1, cv2.LINE_4)
cv2.line(image, (0, 0), (20, 20), colors['red'], 1, cv2.LINE_AA)
cv2.line(image, (0, 5), (15, 20), colors['green'], 1, cv2.LINE_8)

# Show image:
show_with_matplotlib(image, 'LINE_4    LINE_AA    LINE_8')

"""
Example to show how to draw all OpenCV fonts
"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_with_matplotlib(img, title):
    """Shows an image using matplotlib capabilities

    """
    # Convert BGR image to RGB
    img_RGB = img[:, :, ::-1]

    # Show the image using matplotlib:
    plt.imshow(img_RGB)
    plt.title(title)
    plt.show()


# Dictionary containing some colors
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

# Dictionary containing some strings to output
fonts = {0: "FONT HERSHEY SIMPLEX", 1: "FONT HERSHEY PLAIN", 2: "FONT HERSHEY DUPLEX", 3: "FONT HERSHEY COMPLEX",
         4: "FONT HERSHEY TRIPLEX", 5: "FONT HERSHEY COMPLEX SMALL ", 6: "FONT HERSHEY SCRIPT SIMPLEX",
         7: "FONT HERSHEY SCRIPT COMPLEX"}

# Dictionary containing the index for each color
index_colors = {0: 'blue', 1: 'green', 2: 'red', 3: 'yellow', 4: 'magenta', 5: 'cyan', 6: 'black', 7: 'dark_gray'}

# We create the canvas to draw: 650 x 650 pixels, 3 channels, uint8 (8-bit unsigned integers)
# We set background to black using np.zeros()
image = np.zeros((650, 650, 3), dtype="uint8")

# If you want another background color you can do the following:
image[:] = colors['white']

position = (10, 30)
for i in range(0, 8):
    print("i index value: '{}' text: '{}' + color: '{}' = '{}'".format(i, fonts[i].lower(), index_colors[i],
                                                                       colors[index_colors[i]]))
    cv2.putText(image, fonts[i], position, i, 1.1, colors[index_colors[i]], 2, cv2.LINE_4)
    position = (position[0], position[1] + 40)
    cv2.putText(image, fonts[i].lower(), position, i, 1.1, colors[index_colors[i]], 2, cv2.LINE_4)
    position = (position[0], position[1] + 40)

# Show image:
show_with_matplotlib(image, 'cv2.putText() using all OpenCV fonts')

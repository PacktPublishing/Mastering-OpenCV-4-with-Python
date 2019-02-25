"""
Example to show how to draw basic memes with OpenCV
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

# We load the image 'lenna.png':
image = cv2.imread("lenna.png")

# Write some text (up):
cv2.putText(image, 'Hello World', (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.8, colors['green'], 1, cv2.LINE_AA)

# Write some text (down):
cv2.putText(image, 'Goodbye World', (10, 200), cv2.FONT_HERSHEY_TRIPLEX, 0.8, colors['red'], 1, cv2.LINE_AA)

# Show image:
show_with_matplotlib(image, 'very basic meme generator')

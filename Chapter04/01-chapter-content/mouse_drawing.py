"""
Example to show how to draw circles using the mouse events in OpenCV
"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dictionary containing some colors
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}


# This is the mouse callback function:
def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("event: EVENT_LBUTTONDBLCLK")
        cv2.circle(image, (x, y), 10, colors['magenta'], -1)

    if event == cv2.EVENT_MOUSEMOVE:
        print("event: EVENT_MOUSEMOVE")

    if event == cv2.EVENT_LBUTTONUP:
        print("event: EVENT_LBUTTONUP")

    if event == cv2.EVENT_LBUTTONDOWN:
        print("event: EVENT_LBUTTONDOWN")


# We create the canvas to draw: 600 x 600 pixels, 3 channels, uint8 (8-bit unsigned integers)
# We set the background to black using np.zeros()
image = np.zeros((600, 600, 3), dtype="uint8")

# We create a named window where the mouse callback will be established
cv2.namedWindow('Image mouse')

# We set the mouse callback function to 'draw_circle'
cv2.setMouseCallback('Image mouse', draw_circle)

while True:
    # Show image 'Image mouse':
    cv2.imshow('Image mouse', image)

    # Continue until 'q' is pressed
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# Destroy all generated windows:
cv2.destroyAllWindows()

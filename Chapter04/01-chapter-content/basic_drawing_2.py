"""
Example to show how to draw basic shapes using OpenCV (2)
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

# We create the canvas to draw: 300 x 300 pixels, 3 channels, uint8 (8-bit unsigned integers)
# We set background to black using np.zeros()
image = np.zeros((300, 300, 3), dtype="uint8")

# If you want another background color you can do the following:
image[:] = colors['light_gray']

# 1. We are going to see how cv2.clipLine() works
# Draw a rectangle and a line:
cv2.line(image, (0, 0), (300, 300), colors['green'], 3)
cv2.rectangle(image, (0, 0), (100, 100), colors['blue'], 3)

# We call the function cv2.clipLine()
ret, p1, p2 = cv2.clipLine((0, 0, 100, 100), (0, 0), (300, 300))

# cv2.clipLine() returns False if the line is outside the rectangle
# And returns True otherwise
if ret:
    cv2.line(image, p1, p2, colors['yellow'], 3)

# Show image:
show_with_matplotlib(image, 'cv2.clipLine()')

# Clean the canvas to draw again
image[:] = colors['light_gray']

# 2. We are going to see how cv2.arrowedLine() works:
cv2.arrowedLine(image, (50, 50), (200, 50), colors['red'], 3, 8, 0, 0.1)
cv2.arrowedLine(image, (50, 120), (200, 120), colors['green'], 3, cv2.LINE_AA, 0, 0.3)
cv2.arrowedLine(image, (50, 200), (200, 200), colors['blue'], 3, 8, 0, 0.3)

# Show image:
show_with_matplotlib(image, 'cv2.arrowedLine()')

# Clean the canvas to draw again
image[:] = colors['light_gray']

# 3. We are going to see how cv2.ellipse() works:
cv2.ellipse(image, (80, 80), (60, 40), 0, 0, 360, colors['red'], -1)
cv2.ellipse(image, (80, 200), (80, 40), 0, 0, 360, colors['green'], 3)
cv2.ellipse(image, (80, 200), (10, 40), 0, 0, 360, colors['blue'], 3)
cv2.ellipse(image, (200, 200), (10, 40), 0, 0, 180, colors['yellow'], 3)
cv2.ellipse(image, (200, 100), (10, 40), 0, 0, 270, colors['cyan'], 3)
cv2.ellipse(image, (250, 250), (30, 30), 0, 0, 360, colors['magenta'], 3)
cv2.ellipse(image, (250, 100), (20, 40), 45, 0, 360, colors['gray'], 3)

# Show image:
show_with_matplotlib(image, 'cv2.ellipse()')

# Clean the canvas to draw again
image[:] = colors['light_gray']

# 4. We are going to draw several polylines
# These points define a triangle
pts = np.array([[250, 5], [220, 80], [280, 80]], np.int32)
# Reshape to shape (number_vertex, 1, 2)
pts = pts.reshape((-1, 1, 2))
# Print the shapes: this line is not necessary, only for visualization
print("shape of pts '{}'".format(pts.shape))
# Draw this poligon with True option
cv2.polylines(image, [pts], True, colors['green'], 3)

# These points define a triangle
pts = np.array([[250, 105], [220, 180], [280, 180]], np.int32)
# Reshape to shape (number_vertex, 1, 2)
pts = pts.reshape((-1, 1, 2))
# Print the shapes
print("shape of pts '{}'".format(pts.shape))
# Draw this poligon with False option
cv2.polylines(image, [pts], False, colors['green'], 3)

# These points define a pentagon
pts = np.array([[20, 90], [60, 60], [100, 90], [80, 130], [40, 130]], np.int32)
# Reshape to shape (number_vertex, 1, 2)
pts = pts.reshape((-1, 1, 2))
# Print the shapes
print("shape of pts '{}'".format(pts.shape))
# Draw this poligon with True option
cv2.polylines(image, [pts], True, colors['blue'], 3)

# These points define a pentagon
pts = np.array([[20, 180], [60, 150], [100, 180], [80, 220], [40, 220]], np.int32)
# Reshape to shape (number_vertex, 1, 2)
pts = pts.reshape((-1, 1, 2))
# Print the shapes
print("shape of pts '{}'".format(pts.shape))
# Draw this poligon with False option
cv2.polylines(image, [pts], False, colors['blue'], 3)

# These points define a rectangle
pts = np.array([[150, 100], [200, 100], [200, 150], [150, 150]], np.int32)
# Reshape to shape (number_vertex, 1, 2)
pts = pts.reshape((-1, 1, 2))
# Print the shapes
print("shape of pts '{}'".format(pts.shape))
# Draw this poligon with False option
cv2.polylines(image, [pts], True, colors['yellow'], 3)

# These points define a rectangle
pts = np.array([[150, 200], [200, 200], [200, 250], [150, 250]], np.int32)
# Reshape to shape (number_vertex, 1, 2)
pts = pts.reshape((-1, 1, 2))
# Print the shapes
print("shape of pts '{}'".format(pts.shape))
# Draw this poligon with False option
cv2.polylines(image, [pts], False, colors['yellow'], 3)

# Show image:
show_with_matplotlib(image, 'cv2.polylines()')

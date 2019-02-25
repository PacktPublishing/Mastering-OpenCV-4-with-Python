"""
Example to show how to draw text and a bounding box
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

# We create the canvas to draw: 400 x 1200 pixels, 3 channels, uint8 (8-bit unsigned integers)
# We set background to black using np.zeros():
image = np.zeros((400, 1200, 3), dtype="uint8")

# If you want another background color you can do the following:
image[:] = colors['light_gray']

# Assign parameters to be used in the drawing functions:
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 2.5
thickness = 5
text = 'abcdefghijklmnopqrstuvwxyz'
circle_radius = 10

# We get the size of the text:
ret, baseline = cv2.getTextSize(text, font, font_scale, thickness)

# We get the text width and text height from ret:
text_width, text_height = ret

# We center the text in the image:
text_x = int(round((image.shape[1] - text_width) / 2))
text_y = int(round((image.shape[0] + text_height) / 2))

# Draw this point for reference:
cv2.circle(image, (text_x, text_y), circle_radius, colors['green'], -1)

# Draw the rectangle (bounding box of the text):
cv2.rectangle(image, (text_x, text_y + baseline), (text_x + text_width - thickness, text_y - text_height),
              colors['blue'], thickness)

# Draw the circles defining the rectangle:
cv2.circle(image, (text_x, text_y + baseline), circle_radius, colors['red'], -1)
cv2.circle(image, (text_x + text_width - thickness, text_y - text_height), circle_radius, colors['cyan'], -1)

# Draw the baseline line:
cv2.line(image, (text_x, text_y + int(round(thickness / 2))), (text_x + text_width - thickness, text_y +
                                                               int(round(thickness / 2))), colors['yellow'], thickness)
# Write the text centered in the image:
cv2.putText(image, text, (text_x, text_y), font, font_scale, colors['magenta'], thickness)

# Show image:
show_with_matplotlib(image, 'cv2.getTextSize() + cv2.putText()')

"""
Example to show how to capture a double left click with matplotlib events to draw a rectangle
"""

# Import required packages:
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Dictionary containing some colors
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

# We create the canvas to draw: 400 x 400 pixels, 3 channels, uint8 (8-bit unsigned integers)
# We set the background to black using np.zeros()
image = np.zeros((400, 400, 3), dtype="uint8")

# If you want another background color you can do the following:
image[:] = colors['light_gray']


def update_img_with_matplotlib():
    """Updates an image using matplotlib capabilities

    """
    # Convert BGR to RGB image format
    img_RGB = image[:, :, ::-1]

    # Display the image:
    plt.imshow(img_RGB)

    # Redraw the Figure because the image has been updated:
    figure.canvas.draw()


# We define the event listener for the 'button_press_event':
def click_mouse_event(event):
    # Check if a double left click is performed:
    if event.dblclick and event.button == 1:
        # (event.xdata, event.ydata) contains the float coordinates of the mouse click event:
        cv2.rectangle(image, (int(round(event.xdata)), int(round(event.ydata))),
                      (int(round(event.xdata)) + 100, int(round(event.ydata)) + 50), colors['blue'], cv2.FILLED)
    # Call 'update_image()' method to update the Figure:
    update_img_with_matplotlib()


# We create the Figure:
figure = plt.figure()
figure.add_subplot(111)

# To show the image until a click is performed:
update_img_with_matplotlib()

# 'button_press_event' is a MouseEvent where a mouse botton is click (pressed)
# When this event happens the function 'click_mouse_event' is called:
figure.canvas.mpl_connect('button_press_event', click_mouse_event)

# Display the figure:
plt.show()

"""
Example to show how to draw an analog clock OpenCV
"""

# Import required packages:
import cv2
import numpy as np
import datetime
import math


def array_to_tuple(arr):
    return tuple(arr.reshape(1, -1)[0])


# Dictionary containing some colors
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255),
          'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'white': (255, 255, 255), 'black': (0, 0, 0),
          'gray': (125, 125, 125), 'rand': np.random.randint(0, high=256, size=(3,)).tolist(),
          'dark_gray': (50, 50, 50), 'light_gray': (220, 220, 220)}

# We create the canvas to draw: 640 x 640 pixels, 3 channels, uint8 (8-bit unsigned integers)
# We set background to black using np.zeros()
image = np.zeros((640, 640, 3), dtype="uint8")

# If you want another background color you can do the following:
image[:] = colors['light_gray']

# Coordinates to define the origin for the hour markings:
hours_orig = np.array(
    [(620, 320), (580, 470), (470, 580), (320, 620), (170, 580), (60, 470), (20, 320), (60, 170), (169, 61), (319, 20),
     (469, 60), (579, 169)])

# Coordinates to define the destiny for the hour markings:
hours_dest = np.array(
    [(600, 320), (563, 460), (460, 562), (320, 600), (180, 563), (78, 460), (40, 320), (77, 180), (179, 78), (319, 40),
     (459, 77), (562, 179)])

# We draw the hour markings:
for i in range(0, 12):
    cv2.line(image, array_to_tuple(hours_orig[i]), array_to_tuple(hours_dest[i]), colors['black'], 3)

# We draw a big circle, corresponding to the shape of the analog clock
cv2.circle(image, (320, 320), 310, colors['dark_gray'], 8)

# We draw the rectangle containig the text and the text "Mastering OpenCV 4 with Python":
cv2.rectangle(image, (150, 175), (490, 270), colors['dark_gray'], -1)
cv2.putText(image, "Mastering OpenCV 4", (150, 200), 1, 2, colors['light_gray'], 1, cv2.LINE_AA)
cv2.putText(image, "with Python", (210, 250), 1, 2, colors['light_gray'], 1, cv2.LINE_AA)

# We make a copy of the image with the "static" information
image_original = image.copy()

# Now, we draw the "dynamic" information:
while True:
    # Get current date:
    date_time_now = datetime.datetime.now()
    # Get current time from the date:
    time_now = date_time_now.time()
    # Get current hour-minute-second from the time:
    hour = math.fmod(time_now.hour, 12)
    minute = time_now.minute
    second = time_now.second

    print("hour:'{}' minute:'{}' second: '{}'".format(hour, minute, second))

    # Get the hour, minute and second angles:
    second_angle = math.fmod(second * 6 + 270, 360)
    minute_angle = math.fmod(minute * 6 + 270, 360)
    hour_angle = math.fmod((hour * 30) + (minute / 2) + 270, 360)

    print("hour_angle:'{}' minute_angle:'{}' second_angle: '{}'".format(hour_angle, minute_angle, second_angle))

    # Draw the lines corresponding to the hour, minute and second needles
    second_x = round(320 + 310 * math.cos(second_angle * 3.14 / 180))
    second_y = round(320 + 310 * math.sin(second_angle * 3.14 / 180))
    cv2.line(image, (320, 320), (second_x, second_y), colors['blue'], 2)

    minute_x = round(320 + 260 * math.cos(minute_angle * 3.14 / 180))
    minute_y = round(320 + 260 * math.sin(minute_angle * 3.14 / 180))
    cv2.line(image, (320, 320), (minute_x, minute_y), colors['blue'], 8)

    hour_x = round(320 + 220 * math.cos(hour_angle * 3.14 / 180))
    hour_y = round(320 + 220 * math.sin(hour_angle * 3.14 / 180))
    cv2.line(image, (320, 320), (hour_x, hour_y), colors['blue'], 10)

    # Finally, a small circle, corresponding to the point where the three needles joint, is drawn:
    cv2.circle(image, (320, 320), 10, colors['dark_gray'], -1)

    # Show image:
    cv2.imshow("clock", image)

    # We get the image with the static information:
    image = image_original.copy()

    # A wait of 500 milliseconds is performed (to see the displayed image):
    cv2.waitKey(500)

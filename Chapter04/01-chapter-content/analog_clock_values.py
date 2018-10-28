"""
Example to show how to get the values for the hour markings to build the clock
"""

# Import required packages:
import cv2
import numpy as np
import datetime
import math

radius = 300
center = (320, 320)

for x in (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330):
    x_coordinate = center[0] + radius * math.cos(x * 3.14 / 180)
    y_coordinate = center[1] + radius * math.sin(x * 3.14 / 180)
    print("x: {} y: {}".format(round(x_coordinate), round(y_coordinate)))

print("..............")

for x in (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330):
    x_coordinate = center[0] + (radius - 20) * math.cos(x * 3.14 / 180)
    y_coordinate = center[1] + (radius - 20) * math.sin(x * 3.14 / 180)
    print("x: {} y: {}".format(round(x_coordinate), round(y_coordinate)))

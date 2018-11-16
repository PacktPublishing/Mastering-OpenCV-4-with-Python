"""
Example to show how saturation arithmetic work in OpenCV
"""

import numpy as np
import cv2

# There is a difference between OpenCV addition and Numpy addition.
# OpenCV addition is a saturated operation while Numpy addition is a modulo operation.
x = np.uint8([250])
y = np.uint8([50])

# OpenCV addition: values are clipped to ensure they never fall outside the range [0,255]
# 250+50 = 300 => 255:
result_opencv = cv2.add(x, y)
print("cv2.add(x:'{}' , y:'{}') = '{}'".format(x, y, result_opencv))

# NumPy addition: values wrap around
# 250+50 = 300 % 256 = 44:
result_numpy = x + y
print("x:'{}' + y:'{}' = '{}'".format(x, y, result_numpy))

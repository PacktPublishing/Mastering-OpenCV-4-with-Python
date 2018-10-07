"""
Accessing, manipulating pixels in OpenCV (getting and setting methods) with BGR images
"""

# import the necessary packages
import cv2


# Use the function cv2.imread() to read an image.
# The image should be in the working directory or a full path of image should be provided.
# load OpenCV logo image:
img = cv2.imread('logo.png')

# To get the dimensions of the image use img.shape
# img.shape returns a tuple of number of rows, columns and channels (if image is color)
# If image is grayscale, img.shape returns a tuple of number of rows and columns.
# So it can be used to check if loaded image is grayscale or color image.
# Get the shape of the image:
dimensions = img.shape

# Print the dimensions:
print(dimensions)
# This will print '(99, 82, 3)'

# img.shape will return the dimensions of the image in a tuple like this: (99, 82, 3)
# Therefore, we can also get the dimensions in three independent variables.
# Get height, width and the number of channels of the input image
(h, w, c) = img.shape

# Now, we can print these three variables
# Print (h, w, c) values:
print("Dimensions of the image - Height: {}, Width: {}, Channels: {}".format(h, w, c))
# This will print 'Dimensions of the image - Height: 99, Width: 82, Channels: 3'

# Total number of elements is obtained by img.size:
total_number_of_pixels = img.size

# Print the total number of elements:
print("Total number of elements: {}".format(total_number_of_pixels))
# This will print 'Total number of pixels: 24354'

# The total number of pixels is equal to the multiplication of 'height', 'width' and 'channels':
print("Total number of elements: {}".format(h * w * c))
# This will print 'Total number of pixels: 24354'

# Image datatype is obtained by img.dtype.
# img.dtype is very important because a large number of errors is caused by invalid datatype.
# Get the image datatype:
image_dtype = img.dtype

# Print the image datatype:
print("Image datatype: {}".format(image_dtype))
# This should print 'Image datatype: uint8'
# (uint8) = unsigned char

# Use the function cv2.imshow() to show an image in a window.
# The window automatically fits to the image size.
# First argument is the window name.
# Second argument is the image to be displayed.
# Each created window should have different window names.
# Show original image:
cv2.imshow("original image", img)

# cv2.waitKey() is a keyboard binding function.
# The argument is the time in milliseconds.
# The function waits for specified milliseconds for any keyboard event.
# If any key is pressed in that time, the program continues.
# If 0 is passed, it waits indefinitely for a key stroke.
# Wait indefinitely for a key stroke (in order to see the created window):
cv2.waitKey(0)

# You can access a pixel value by row and column coordinates.
# For BGR image, it returns an array of (Blue, Green, Red) values.
# Get the value of the pixel (x=40, y=6):
(b, g, r) = img[6, 40]

# Print the values:
print("Pixel at (6,40) - Red: {}, Green: {}, Blue: {}".format(r, g, b))
# This will print 'Pixel at (6,40) - Red: 247, Green: 18, Blue: 36'

# We can access only one channel at a time.
# In this case, we will use row, column and the index of the desired channel for indexing.
# Get only blue value of the pixel (x=40, y=6):
b = img[6, 40, 0]

# Get only green value of the pixel (x=40, y=6):
g = img[6, 40, 1]

# Get only red value of the pixel (x=40, y=6):
r = img[6, 40, 2]

# Print the values again:
print("Pixel at (6,40) - Red: {}, Green: {}, Blue: {}".format(r, g, b))
# This will print 'Pixel at (6,40) - Red: 247, Green: 18, Blue: 36'

# You can modify the pixel values of the image in the same way.
# Set the pixel to red ((b - g - r) format):
img[6, 40] = (0, 0, 255)

# Get the value of the pixel (x=40, y=6) after modifying it
(b, g, r) = img[6, 40]

# Print it:
print("Pixel at (6,40) - Red: {}, Green: {}, Blue: {}".format(r, g, b))
# This will print 'Pixel at (6,40) - Red: 255, Green: 0, Blue: 0'

# Sometimes, you will have to play with certain region of images rather than one pixel at a time
# In this case, we get the top left corner of the image:
top_left_corner = img[0:50, 0:50]

# We show this ROI:
cv2.imshow("top left corner original", top_left_corner)

# Wait indefinitely for a key stroke (in order to see the created window):
cv2.waitKey(0)

# Copy this ROI into another region of the image:
img[20:70, 20:70] = top_left_corner

# We show the modified image:
cv2.imshow("modified image", img)

# Wait indefinitely for a key stroke (in order to see the created window):
cv2.waitKey(0)

# Set top left corner of the image to blue
img[0:50, 0:50] = (255, 0, 0)

# Show modified image;
cv2.imshow("modified image", img)

# Wait indefinitely for a key stroke (in order to see the created windows):
cv2.waitKey(0)

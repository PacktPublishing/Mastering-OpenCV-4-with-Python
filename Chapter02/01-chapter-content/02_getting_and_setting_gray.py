"""
Accessing, manipulating pixels in OpenCV (getting and setting methods) with grayscale images
"""

# import the necessary packages
import cv2


# Use the function cv2.imread() to read an image.
# The image should be in the working directory or a full path of image should be provided.
# In this case, the second argument is needed, because we want to load the image in grayscale.
# Second argument is a flag which specifies the way image should be read.
# Value needed for loading a image in grayscale: 'cv2.IMREAD_GRAYSCALE'.
# load OpenCV logo image:
gray_img = cv2.imread('logo.png', cv2.IMREAD_GRAYSCALE)

# To get the dimensions of the image use img.shape
# img.shape returns a tuple of number of rows, columns and channels (if image is color)
# If image is grayscale, img.shape returns a tuple of number of rows and columns.
# So it can be used to check if loaded image is grayscale or color image.
# Get the shape of the image (in this case only two components!):
dimensions = gray_img.shape

# Print the dimensions:
print(dimensions)
# This will print '(99, 82)'

# img.shape will return the dimensions of the image in a tuple like this: (99, 82)
# Therefore, we can also get the dimensions in two independent variables.
# Get height and width of the input grayscale image
(h, w) = gray_img.shape

# Now, we can print these two variables
# Print (h, w) values:
print("Dimensions of the image - Height: {}, Width: {}".format(h, w))
# This will print 'Dimensions of the image - Height: 99, Width: 82'

# Total number of elements is obtained by img.size:
total_number_of_pixels = gray_img.size

# Print the total number of elements:
print("Total number of elements: {}".format(total_number_of_pixels))
# This will print 'Total number of elements: 8118'

# The total number of pixels is equal to the multiplication of 'height', 'width' and 'channels':
print("Total number of elements: {}".format(h * w))
# This will print 'Total number of elements: 8118'

# Image datatype is obtained by img.dtype.
# img.dtype is very important because a large number of errors is caused by invalid datatype.
# Get the image datatype:
image_dtype = gray_img.dtype

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
cv2.imshow("original image", gray_img)

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
i = gray_img[6, 40]

# Print the value:
print("Pixel at (6,40) - Intensity: {}".format(i))
# This will print 'Pixel at (6,40) - Intensity: 88'

# You can modify the pixel values of the image in the same way.
# Set the pixel to black:
gray_img[6, 40] = 0

# Get the value of the pixel (x=40, y=6) after modifying it
i = gray_img[6, 40]

# Print the value:
print("Pixel at (6,40) - Intensity: {}".format(i))
# This will print 'Pixel at (6,40) - Intensity: 0'

# Sometimes, you will have to play with certain region of images rather than one pixel at a time
# In this case, we get the top left corner of the image:
top_left_corner = gray_img[0:50, 0:50]

# We show this ROI:
cv2.imshow("top left corner original", top_left_corner)

# Wait indefinitely for a key stroke (in order to see the created window):
cv2.waitKey(0)

# Copy this ROI into another region of the image:
gray_img[20:70, 20:70] = top_left_corner

# We show the modified image:
cv2.imshow("modified image", gray_img)

# Wait indefinitely for a key stroke (in order to see the created window):
cv2.waitKey(0)

# Set top left corner of the image to white
gray_img[0:50, 0:50] = 255

# Show modified image;
cv2.imshow("modified image", gray_img)

# Wait indefinitely for a key stroke (in order to see the created windows):
cv2.waitKey(0)

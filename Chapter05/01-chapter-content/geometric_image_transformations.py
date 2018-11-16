"""
Geometric image transformations
"""

# Import required packages:
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_with_matplotlib(img, title):
    """Shows an image using matplotlib capabilities"""

    # Convert BGR image to RGB
    img_RGB = img[:, :, ::-1]

    # Show the image using matplotlib:
    plt.imshow(img_RGB)
    plt.title(title)
    plt.show()


# Read the input image:
image = cv2.imread('lena_image.png')

# Show loaded image:
show_with_matplotlib(image, 'Original image')

# 1. Scaling or resizing
# Resize the input image using cv2.resize()
# Resize using the scaling factor for each dimension of the image
# In this case the scaling factor is 0.5 in every dimension
dst_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

# Get the height and width of the image:
height, width = image.shape[:2]

# You can resize also the image specifying the new size:
dst_image_2 = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

# We see the two resized images:
show_with_matplotlib(dst_image, 'Resized image')
show_with_matplotlib(dst_image_2, 'Resized image 2')

# 2. Translation
# You need to create the 2x3 transformation matrix making use of numpy array with float values (float32)
# Translation in the x direction: 200 pixels, and a translation in the y direction: 30 pixels:
M = np.float32([[1, 0, 200], [0, 1, 30]])

# Once this transformation Matrix is created, we can pass it to the function cv2.warpAffine():
dst_image = cv2.warpAffine(image, M, (width, height))

# Show the translated image:
show_with_matplotlib(dst_image, 'Translated image (positive values)')

# Additionally, the translation can take negative values:
M = np.float32([[1, 0, -200], [0, 1, -30]])
dst_image = cv2.warpAffine(image, M, (width, height))

# Show the translated image:
show_with_matplotlib(dst_image, 'Translated image (negative values)')

# 3. Rotation
# To rotate the image we make use of the function  cv.getRotationMatrix2D() to build the 2x3 rotation matrix:
# In this case, we are going to rotate the image 180 degrees (upside down):
M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 180, 1)
dst_image = cv2.warpAffine(image, M, (width, height))

# Show the center of rotation and the rotated image:
cv2.circle(dst_image, (round(width / 2.0), round(height / 2.0)), 5, (255, 0, 0), -1)
show_with_matplotlib(dst_image, 'Image rotated 180 degrees')

# Now, we are going to rotate the image 30 degrees changing the center of rotation
M = cv2.getRotationMatrix2D((width / 1.5, height / 1.5), 30, 1)
dst_image = cv2.warpAffine(image, M, (width, height))

# Show the center of rotation and the rotated image:
cv2.circle(dst_image, (round(width / 1.5), round(height / 1.5)), 5, (255, 0, 0), -1)
show_with_matplotlib(dst_image, 'Image rotated 30 degrees')

# 4. Affine Transformation
# In an affine transformation we first make use of the function cv2.getAffineTransform()
# to build the 2x3 transformation matrix, which is obtained from the relation between three points
# from the input image and their corresponding coordinates in the transformed image.

# A copy of the image is created to show the points that will be used for the affine transformation:
image_points = image.copy()
cv2.circle(image_points, (135, 45), 5, (255, 0, 255), -1)
cv2.circle(image_points, (385, 45), 5, (255, 0, 255), -1)
cv2.circle(image_points, (135, 230), 5, (255, 0, 255), -1)

# Show the image with the three created points:
show_with_matplotlib(image_points, 'before affine transformation')

# We create the arrays with the aforementioned three points and the desired positions in the output image:
pts_1 = np.float32([[135, 45], [385, 45], [135, 230]])
pts_2 = np.float32([[135, 45], [385, 45], [150, 230]])

# We get the 2x3 tranformation matrix based on pts_1 and pts_2 and apply cv2.warpAffine():
M = cv2.getAffineTransform(pts_1, pts_2)
dst_image = cv2.warpAffine(image_points, M, (width, height))

# Show the image:
show_with_matplotlib(dst_image, 'Affine transformation')

# 5. Perspective transformation
# A copy of the image is created to show the points that will be used for the perspective transformation:
image_points = image.copy()
cv2.circle(image_points, (450, 65), 5, (255, 0, 255), -1)
cv2.circle(image_points, (517, 65), 5, (255, 0, 255), -1)
cv2.circle(image_points, (431, 164), 5, (255, 0, 255), -1)
cv2.circle(image_points, (552, 164), 5, (255, 0, 255), -1)

# Show the image:
show_with_matplotlib(image_points, 'before perspective transformation')

# cv2.getPerspectiveTransform() needs four pairs of points
# (coordinates of a quadrangle in both the source and output image)
# We create the arrays for these four pairs of points:
pts_1 = np.float32([[450, 65], [517, 65], [431, 164], [552, 164]])
pts_2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

# To correct the perspective (also known as perspective transformation) you need to create the transformation matrix
# making use of the function cv2.getPerspectiveTransform(), where a 3x3 matrix is constructed:
M = cv2.getPerspectiveTransform(pts_1, pts_2)

# Then, apply cv2.warpPerspective(), where the source image is transformed applying
# the specified matrix and with a specified size:
dst_image = cv2.warpPerspective(image, M, (300, 300))

# Show the image:
show_with_matplotlib(dst_image, 'perspective transformation')

# 6. Cropping
# A copy of the image is created to show the points that will be used for the cropping example:
image_points = image.copy()

# Show the points and lines connecting the points:
cv2.circle(image_points, (230, 80), 5, (0, 0, 255), -1)
cv2.circle(image_points, (330, 80), 5, (0, 0, 255), -1)
cv2.circle(image_points, (230, 200), 5, (0, 0, 255), -1)
cv2.circle(image_points, (330, 200), 5, (0, 0, 255), -1)
cv2.line(image_points, (230, 80), (330, 80), (0, 0, 255))
cv2.line(image_points, (230, 200), (330, 200), (0, 0, 255))
cv2.line(image_points, (230, 80), (230, 200), (0, 0, 255))
cv2.line(image_points, (330, 200), (330, 80), (0, 0, 255))

# Show the image with the points and lines:
show_with_matplotlib(image_points, 'Before cropping')

# For cropping, we make use of numpy slicing:
dst_image = image[80:200, 230:330]

# Show the image:
show_with_matplotlib(dst_image, 'Cropping image')

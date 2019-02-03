"""
Understanding cv2.dnn.blobFromImage() and also cv2.dnn.imagesFromBlob() in OpenCV
"""

# Import required packages:
import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 4, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def get_image_from_blob(blob_img, scalefactor, dim, mean, swap_rb, mean_added):
    """Returns image from blob assuming that the blob is from only one image"""

    images_from_blob = cv2.dnn.imagesFromBlob(blob_img)
    image_from_blob = np.reshape(images_from_blob[0], dim) / scalefactor
    image_from_blob_mean = np.uint8(image_from_blob)
    image_from_blob = image_from_blob_mean + np.uint8(mean)

    if mean_added is True:
        if swap_rb:
            image_from_blob = image_from_blob[:, :, ::-1]
        return image_from_blob
    else:
        if swap_rb:
            image_from_blob_mean = image_from_blob_mean[:, :, ::-1]
        return image_from_blob_mean


# Load image:
image = cv2.imread("face_test.jpg")

# Call cv2.dnn.blobFromImage():
blob_image = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104., 117., 123.], False, False)

# The shape of the blob_image will be (1, 3, 300, 300):
print(blob_image.shape)

# Get different images from the blob:
img_from_blob = get_image_from_blob(blob_image, 1.0, (300, 300, 3), [104., 117., 123.], False, True)
img_from_blob_swap = get_image_from_blob(blob_image, 1.0, (300, 300, 3), [104., 117., 123.], True, True)
img_from_blob_mean = get_image_from_blob(blob_image, 1.0, (300, 300, 3), [104., 117., 123.], False, False)
img_from_blob_mean_swap = get_image_from_blob(blob_image, 1.0, (300, 300, 3), [104., 117., 123.], True, False)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(16, 4))
plt.suptitle("cv2.dnn.blobFromImage() visualization", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Show the created images:
show_img_with_matplotlib(img_from_blob, "img from blob " + str(img_from_blob.shape), 1)
show_img_with_matplotlib(img_from_blob_swap, "img from blob swap " + str(img_from_blob.shape), 2)
show_img_with_matplotlib(img_from_blob_mean, "img from blob mean " + str(img_from_blob.shape), 3)
show_img_with_matplotlib(img_from_blob_mean_swap, "img from blob mean swap " + str(img_from_blob.shape), 4)

# Show the Figure:
plt.show()

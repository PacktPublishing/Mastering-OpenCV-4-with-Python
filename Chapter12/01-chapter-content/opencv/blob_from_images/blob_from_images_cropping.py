"""
Understanding cv2.dnn.blobFromImages() and cv2.dnn.imagesFromBlob() in OpenCV with cropping
"""

# Import required packages:
import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_cropped_img(img):
    """Returns the cropped image"""

    # Create a copy of the image:
    img_copy = img.copy()

    # calculate size of resulting image:
    size = min(img_copy.shape[1], img_copy.shape[0])

    # calculate x1, and y1
    x1 = int(0.5 * (img_copy.shape[1] - size))
    y1 = int(0.5 * (img_copy.shape[0] - size))

    # crop and return the image
    return img_copy[y1:(y1 + size), x1:(x1 + size)]


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def get_images_from_blob(blob_imgs, scalefactor, dim, mean, swap_rb, mean_added):
    """Returns images from blob"""

    images_from_blob = cv2.dnn.imagesFromBlob(blob_imgs)
    imgs = []

    for image_blob in images_from_blob:
        image_from_blob = np.reshape(image_blob, dim) / scalefactor
        image_from_blob_mean = np.uint8(image_from_blob)
        image_from_blob = image_from_blob_mean + np.uint8(mean)
        if mean_added is True:
            if swap_rb:
                image_from_blob = image_from_blob[:, :, ::-1]
            imgs.append(image_from_blob)
        else:
            if swap_rb:
                image_from_blob_mean = image_from_blob_mean[:, :, ::-1]
            imgs.append(image_from_blob_mean)

    return imgs


# Load images and get the list of images:
image = cv2.imread("face_test.jpg")
image2 = cv2.imread("face_test_2.jpg")
images = [image, image2]

# To see how cropping works, we are going to perform the cropping formulation that
# both blobFromImage() and blobFromImages() perform applying it to one of the input images:
cropped_img = get_cropped_img(image)
# cv2.imwrite("cropped_img.jpg", cropped_img)

# Call cv2.dnn.blobFromImages():
blob_images = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, False)
blob_blob_images_cropped = cv2.dnn.blobFromImages(images, 1.0, (300, 300), [104., 117., 123.], False, True)

# Get different images from the blob:
imgs_from_blob = get_images_from_blob(blob_images, 1.0, (300, 300, 3), [104., 117., 123.], False, True)
imgs_from_blob_cropped = get_images_from_blob(blob_blob_images_cropped, 1.0, (300, 300, 3), [104., 117., 123.], False,
                                              True)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 8))
plt.suptitle("cv2.dnn.blobFromImages() visualization with cropping", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Show the input images
show_img_with_matplotlib(imgs_from_blob[0], "img 1 from blob " + str(imgs_from_blob[0].shape), 1)
show_img_with_matplotlib(imgs_from_blob[1], "img 2 from blob " + str(imgs_from_blob[1].shape), 2)
show_img_with_matplotlib(imgs_from_blob_cropped[0], "img 1 from blob cropped " + str(imgs_from_blob[1].shape), 3)
show_img_with_matplotlib(imgs_from_blob_cropped[1], "img 2 from blob cropped " + str(imgs_from_blob[1].shape), 4)

# Show the Figure:
plt.show()

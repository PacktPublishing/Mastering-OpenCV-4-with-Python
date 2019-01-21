"""
Cat face detection using haar feature-based cascade classifiers
"""

# Import required packages:
import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_detection(image, faces):
    """Draws a rectangle over each detected face"""

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 5)
    return image


# Load image and convert to grayscale:
img = cv2.imread("test_cat_face_detection.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load cascade classifiers:
cas_catface = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
cas_catface_extended = cv2.CascadeClassifier("haarcascade_frontalcatface_extended.xml")

# Detect faces:
faces_cas_catface = cas_catface.detectMultiScale(gray)
faces_cas_catface_extended = cas_catface_extended.detectMultiScale(gray)
retval, faces_haar_cat = cv2.face.getFacesHAAR(img, "haarcascade_frontalcatface.xml")
faces_haar_cat = np.squeeze(faces_haar_cat)
retval, faces_haar_cat_extended = cv2.face.getFacesHAAR(img, "haarcascade_frontalcatface_extended.xml")
faces_haar_cat_extended = np.squeeze(faces_haar_cat_extended)

# Draw cat face detections:
img_cas_catface = show_detection(img.copy(), faces_cas_catface)
img_cas_catface_extended = show_detection(img.copy(), faces_cas_catface_extended)
img_faces_haar_cat = show_detection(img.copy(), faces_haar_cat)
img_faces_haar_cat_extended = show_detection(img.copy(), faces_haar_cat_extended)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 8))
plt.suptitle("Cat face detection using haar feature-based cascade classifiers", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_cas_catface, "detectMultiScale(frontalcatface): " + str(len(faces_cas_catface)), 1)
show_img_with_matplotlib(img_cas_catface_extended,
                         "detectMultiScale(frontalcatface_extended): " + str(len(faces_cas_catface_extended)), 2)
show_img_with_matplotlib(img_faces_haar_cat, "getFacesHAAR(frontalcatface): " + str(len(faces_haar_cat)), 3)
show_img_with_matplotlib(img_faces_haar_cat_extended,
                         "getFacesHAAR(frontalcatface_extended): " + str(len(faces_haar_cat_extended)), 4)

# Show the Figure:
plt.show()

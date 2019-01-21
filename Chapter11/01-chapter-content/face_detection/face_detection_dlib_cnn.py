"""
Face detection using dlib CNN face detector using a pre-trained model (712 KB)
from http://dlib.net/files/mmod_human_face_detector.dat.bz2.
"""

# Import required packages:
import cv2
import dlib
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_detection(image, faces):
    """Draws a rectangle over each detected face"""

    # faces contains a list of mmod_rectangle objects
    # The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score
    # Therefore, we iterate over the detected mmod_rectangle objects accessing dlib.rect to draw the rectangle

    for face in faces:
        cv2.rectangle(image, (face.rect.left(), face.rect.top()), (face.rect.right(), face.rect.bottom()), (255, 0, 0),
                      10)
    return image


# Load CNN detector from dlib:
cnn_face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")

# Load image and convert to grayscale:
img = cv2.imread("test_face_detection.jpg")

# Resize the image to attain reasonable speed:
# img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

# Detect faces:
rects = cnn_face_detector(img, 0)

# Draw face detections:
img_faces = show_detection(img.copy(), rects)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 5))
plt.suptitle("Face detection using dlib CNN face detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_faces, "cnn_face_detector(img, 0): " + str(len(rects)), 1)

# Show the Figure:
plt.show()

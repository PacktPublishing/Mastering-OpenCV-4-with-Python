"""
Face detection using dlib frontal face detector, which is based on Histogram of Oriented Gradients (HOG) features
and a linear classifier in a sliding window detection approach
"""

# Import required packages:
import cv2
import dlib
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


def show_detection(image, faces):
    """Draws a rectangle over each detected face"""

    for face in faces:
        cv2.rectangle(image, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 10)
    return image


# Load image and convert to grayscale:
img = cv2.imread("test_face_detection.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Load frontal face detector from dlib:
detector = dlib.get_frontal_face_detector()

# Detect faces:
rects_1 = detector(gray, 0)
rects_2 = detector(gray, 1)

# Draw face detections:
img_faces_1 = show_detection(img.copy(), rects_1)
img_faces_2 = show_detection(img.copy(), rects_2)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 4))
plt.suptitle("Face detection using dlib frontal face detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_faces_1, "detector(gray, 0): " + str(len(rects_1)), 1)
show_img_with_matplotlib(img_faces_2, "detector(gray, 1): " + str(len(rects_2)), 2)

# Show the Figure:
plt.show()

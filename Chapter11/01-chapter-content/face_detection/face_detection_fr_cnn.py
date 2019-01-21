"""
Face detection using face_recognition CNN face detector (internally calls dlib CNN face detector)
"""

# Import required packages:
import cv2
import face_recognition
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
        top, right, bottom, left = face
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 3)
    return image


# Load image and resize:
img = cv2.imread("test_face_detection.jpg")
img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)

# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
rgb = img[:, :, ::-1]

# Perform face detection using face_recognition (internally using dlib CNN face detection):
rects_1 = face_recognition.face_locations(rgb, 0, "cnn")
rects_2 = face_recognition.face_locations(rgb, 1, "cnn")

# Draw face detections:
img_faces_1 = show_detection(img.copy(), rects_1)
img_faces_2 = show_detection(img.copy(), rects_2)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 4))
plt.suptitle("Face detection using face_recognition CNN face detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(img_faces_1, "face_locations(rgb, 0, cnn): " + str(len(rects_1)), 1)
show_img_with_matplotlib(img_faces_2, "face_locations(rgb, 1, cnn): " + str(len(rects_2)), 2)

# Show the Figure:
plt.show()

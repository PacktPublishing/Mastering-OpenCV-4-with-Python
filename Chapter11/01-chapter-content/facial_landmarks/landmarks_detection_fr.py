"""
Detecting facial landmarks using face_recognition
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


# Load image:
image = cv2.imread("face_test.png")

# Create images to show the results:
image_68 = image.copy()
image_5 = image.copy()

# Convert the image from BGR color (which OpenCV uses) to RGB color:
rgb = image[:, :, ::-1]

# Detect 68 landmarks:
face_landmarks_list_68 = face_recognition.face_landmarks(rgb)

# Print detected landmarks:
print(face_landmarks_list_68)

# Draw all detected landmarks:
for face_landmarks in face_landmarks_list_68:
    for facial_feature in face_landmarks.keys():
        for p in face_landmarks[facial_feature]:
            cv2.circle(image_68, p, 2, (0, 255, 0), -1)

# Detect 5 landmarks:
face_landmarks_list_5 = face_recognition.face_landmarks(rgb, None, "small")

# Print detected landmarks:
print(face_landmarks_list_5)

# Draw all detected landmarks:
for face_landmarks in face_landmarks_list_5:
    for facial_feature in face_landmarks.keys():
        for p in face_landmarks[facial_feature]:
            cv2.circle(image_5, p, 2, (0, 255, 0), -1)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 5))
plt.suptitle("Facial landmarks detection using face_recognition", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot the images:
show_img_with_matplotlib(image_68, "68 facial landmarks", 1)
show_img_with_matplotlib(image_5, "5 facial landmarks", 2)

# Show the Figure:
plt.show()

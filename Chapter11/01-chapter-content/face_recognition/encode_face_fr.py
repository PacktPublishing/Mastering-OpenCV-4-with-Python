"""
This script makes used of face_recognition library to calculate the 128D descriptor to be used for face recognition.
"""

# Import required packages:
import face_recognition
import cv2

# Load image:
image = cv2.imread("jared_1.jpg")

# Convert image from BGR (OpenCV format) to RGB (face_recognition format):
image = image[:, :, ::-1]

# Calculate the encodings for every face of the image:
encodings = face_recognition.face_encodings(image)

# Show the first encoding:
print(encodings[0])

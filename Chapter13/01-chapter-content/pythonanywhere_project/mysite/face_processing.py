#Its Amazing Project
# Import required packages:
import cv2
import numpy as np


class FaceProcessing(object):
    def __init__(self):
        self.file = "/home/opencv/mysite/haarcascade_frontalface_alt.xml"
        self.face_cascade = cv2.CascadeClassifier(self.file)

    def face_detection(self, image):
        # Convert image to OpenCV format:
        image_array = np.asarray(bytearray(image), dtype=np.uint8)
        img_opencv = cv2.imdecode(image_array, -1)
        output = []
        # Detect faces and build output:
        gray = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
        for face in faces:
            # face.tolist(): returns a copy of the array data as a Python list
            x, y, w, h = face.tolist()
            face = {"box": [x, y, x + w, y + h]}
            output.append(face)
        # Return output:
        return output

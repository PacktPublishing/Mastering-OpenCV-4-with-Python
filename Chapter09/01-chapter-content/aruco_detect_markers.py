"""
This script detects markers using Aruco from the webcam
"""

# Import required packages:
import cv2

# We create the dictionary object:
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

# We create the parameters object:
parameters = cv2.aruco.DetectorParameters_create()

# Create video capture object 'capture' to be used to capture frames from the first connected camera:
capture = cv2.VideoCapture(0)

while True:
    # Capture frame by frame from the video capture object 'capture':
    ret, frame = capture.read()

    # We convert the frame to grayscale:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # We call the function 'cv2.aruco.detectMarkers()'
    # The first parameter is the image where the markers are going to be detected
    # The second parameter is the dictionary object
    # The third parameter establishes all the parameters that can be customized during the detection process
    # Returns list of corners and identifiers of the detected markers and list of rejected candidates
    corners, ids, rejected_corners = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)

    # Draw detected markers:
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))

    # Draw rejected markers:
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejected_corners, borderColor=(0, 0, 255))

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything:
capture.release()
cv2.destroyAllWindows()

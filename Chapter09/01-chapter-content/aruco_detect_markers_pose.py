"""
This script detects markers using Aruco from the webcam and draw pose
"""

# Import required packages
import cv2
import os
import pickle

# Check for camera calibration data
if not os.path.exists('./calibration.pckl'):
    print("You need to calibrate the camera you'll be using. See calibration script.")
    exit()
else:
    f = open('calibration.pckl', 'rb')
    cameraMatrix, distCoeffs = pickle.load(f)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print("Calibration issue. Remove ./calibration.pckl and recalibrate your camera")
        exit()

# We create the dictionary object. Aruco has some predefined dictionaries.
# (DICT_4X4_100, DICT_4X4_1000, DICT_4X4_250, DICT_4X4_50 = 0, .... , DICT_7X7_1000)
# We are going to create a dictionary, which is composed by 250 markers.
# Each marker will be of 5x5 bits (DICT_7X7_250):
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

# Create parameters to be used when detecting markers:
parameters = cv2.aruco.DetectorParameters_create()

# Create video capture object 'capture' to be used to capture frames from the first connected camera:
capture = cv2.VideoCapture(0)

while True:
    # Capture frame by frame from the video capture object 'capture':
    ret, frame = capture.read()

    # We convert the frame to grayscale:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # lists of ids and the corners beloning to each id
    # We call the function 'cv2.aruco.detectMarkers()'
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)

    # Draw detected markers:
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))

    # Draw rejected markers:
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejectedImgPoints, borderColor=(0, 0, 255))

    if ids is not None:
        # rvecs and tvecs are the rotation and translation vectors respectively, for each of the markers in corners.
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 1)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything:
capture.release()
cv2.destroyAllWindows()

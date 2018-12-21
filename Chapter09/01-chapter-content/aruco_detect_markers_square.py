"""
This script detects markers using Aruco from the webcam and overlays a cyan square
"""

# Import required packages
import cv2
import os
import pickle
import numpy as np

OVERLAY_SIZE_PER = 1

# Check for camera calibration data
if not os.path.exists('./calibration.pckl'):
    print("You need to calibrate the camera before")
    exit()
else:
    f = open('calibration.pckl', 'rb')
    cameraMatrix, distCoeffs = pickle.load(f)
    f.close()
    if cameraMatrix is None or distCoeffs is None:
        print("Something went wrong. Recalibrate the camera")
        exit()


def draw_points(img, pts):
    """ Draw the points in the image"""

    pts = np.int32(pts).reshape(-1, 2)

    img = cv2.drawContours(img, [pts], -1, (255, 255, 0), -3)

    for p in pts:
        cv2.circle(img, (p[0], p[1]), 5, (255, 0, 255), -1)

    return img


# Create the dictionary object and the parameters:
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)
parameters = cv2.aruco.DetectorParameters_create()

# Create video capture object 'capture' to be used to capture frames from the first connected camera:
capture = cv2.VideoCapture(0)

while True:
    # Capture frame by frame from the video capture object 'capture':
    ret, frame = capture.read()

    # We convert the frame to grayscale:
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers:
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)

    # Draw detected markers:
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))

    # Draw rejected markers:
    # frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejectedImgPoints, borderColor=(0, 0, 255))

    if ids is not None:
        # rvecs and tvecs are the rotation and translation vectors respectively, for each of the markers in corners.
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)

        for rvec, tvec in zip(rvecs, tvecs):
            # Note: The marker coordinate system is centered on the center of the marker
            # The coordinates of the four corners of the marker (in its own coordinate system) are:
            # 1: (-markerLength/2, markerLength/2, 0)
            # 2: (markerLength/2, markerLength/2, 0)
            # 3: (markerLength/2, -markerLength/2, 0)
            # 4: (-markerLength/2, -markerLength/2, 0)
            # Define the points where you want the image to be overlaid (remember: marker coordinate system):
            desired_points = np.float32(
                [[-1 / 2, 1 / 2, 0], [1 / 2, 1 / 2, 0], [1 / 2, -1 / 2, 0], [-1 / 2, -1 / 2, 0]]) * OVERLAY_SIZE_PER

            # Project the points:
            projected_desired_points, jac = cv2.projectPoints(desired_points, rvecs, tvecs, cameraMatrix, distCoeffs)

            # Draw the projected points (debugging):
            draw_points(frame, projected_desired_points)

    # Display the resulting augmented frame:
    cv2.imshow('frame', frame)

    # Press 'q' to exit:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything:
capture.release()
cv2.destroyAllWindows()

"""
Example to introduce how to read a video file
"""

# Import the required packages
import cv2
import numpy as np
import argparse

# We first create the ArgumentParser object
# The created object 'parser' will have the necessary information
# to parse the command-line arguments into data types.
parser = argparse.ArgumentParser()

# We add 'video_path' argument using add_argument() including a help.
parser.add_argument("video_path", help="path to the video file")
args = parser.parse_args()

# Create a VideoCapture object. In this case, the argument is the video file name:
capture = cv2.VideoCapture(args.video_path)
 
# Check if the video is opened successfully
if capture.isOpened()is False:
    print("Error opening the video file!")
 
# Read until video is completed, or 'q' is pressed
while capture.isOpened():
    # Capture frame-by-frame from the video file
    ret, frame = capture.read()

    if ret is True:
        # Display the resulting frame
        cv2.imshow('Original frame from the video file', frame)

        # Convert the frame from the video file to grayscale:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the grayscale frame
        cv2.imshow('Grayscale frame', gray_frame)
 
        # Press q on keyboard to exit the program
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
 
# Release everything
capture.release()
cv2.destroyAllWindows()

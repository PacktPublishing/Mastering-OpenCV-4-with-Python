"""
Example to introduce how to read a IP camera
"""

# Import the required packages
import cv2
import argparse

# We first create the ArgumentParser object
# The created object 'parser' will have the necessary information
# to parse the command-line arguments into data types.
parser = argparse.ArgumentParser()

# We add 'ip_url' argument using add_argument() including a help.
parser.add_argument("ip_url", help="IP URL to connect")
args = parser.parse_args()

# Create a VideoCapture object
# In this case, the argument is the URL connection of the IP camera
# cap = cv2.VideoCapture("http://217.126.89.102:8010/axis-cgi/mjpg/video.cgi")
capture = cv2.VideoCapture(args.ip_url)

# Check if camera opened successfully
if capture.isOpened()is False:
    print("Error opening the camera")
 
# Read until the video is completed, or 'q' is pressed
while capture.isOpened():
    # Capture frame-by-frame from the camera
    ret, frame = capture.read()

    if ret is True:
        # Display the captured frame:
        cv2.imshow('Club Nàutic Port de la Selva (Girona - Spain)', frame)

        # Convert the frame captured from the camera to grayscale:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the grayscale frame
        cv2.imshow('Grayscale Club Nàutic Port de la Selva (Girona - Spain)', gray_frame)

        # Press q on keyboard to exit the program
        if cv2.waitKey(2000) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
 
# Release everything:
capture.release()
cv2.destroyAllWindows()

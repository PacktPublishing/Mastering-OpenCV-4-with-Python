"""
Example to introduce how to read a video file backwards and save it
"""

# Import the required packages
import cv2
import argparse

def decode_fourcc(fourcc):
    """Decodes the fourcc value to get the four chars identifying it

    """
    fourcc_int = int(fourcc)

    # We print the int value of fourcc
    print("int value of fourcc: '{}'".format(fourcc_int))

    # We can also perform this in one line:
    # return "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

    fourcc_decode = ""
    for i in range(4):
        int_value = fourcc_int >> 8 * i & 0xFF
        print("int_value: '{}'".format(int_value))
        fourcc_decode += chr(int_value)
    return fourcc_decode


# We first create the ArgumentParser object
# The created object 'parser' will have the necessary information
# to parse the command-line arguments into data types.
parser = argparse.ArgumentParser()

# We add 'video_path' argument using add_argument() including a help.
parser.add_argument("video_path", help="path to the video file")

# We add 'output_video_path' argument using add_argument() including a help.
parser.add_argument("output_video_path", help="path to the video file to write")

args = parser.parse_args()

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
capture = cv2.VideoCapture(args.video_path)

# Get some properties of VideoCapture (frame width, frame height and frames per second (fps)):
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)
codec = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))

print("codec: '{}'".format(codec))

# FourCC is a 4-byte code used to specify the video codec and it is platform dependent!
fourcc = cv2.VideoWriter_fourcc(*codec)


# Create VideoWriter object. We use the same properties as the input camera.
# Last argument is False to write the video in grayscale. True otherwise (write the video in color)
out = cv2.VideoWriter(args.output_video_path, fourcc, int(fps), (int(frame_width), int(frame_height)), True)

# Check if camera opened successfully
if capture.isOpened()is False:
    print("Error opening video stream or file")

# We get the index of the last frame of the video file
frame_index = capture.get(cv2.CAP_PROP_FRAME_COUNT) - 1
# print("starting in frame: '{}'".format(frame_index))

# Read until video is completed
while capture.isOpened() and frame_index >= 0:

    # We set the current frame position
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

    # Capture frame-by-frame from the video file:
    ret, frame = capture.read()

    if ret is True:

        # Print current frame number per iteration
        # print("CAP_PROP_POS_FRAMES : '{}'".format(capture.get(cv2.CAP_PROP_POS_FRAMES)))

        # Get the timestamp of the current frame in milliseconds
        # print("CAP_PROP_POS_MSEC : '{}'".format(capture.get(cv2.CAP_PROP_POS_MSEC)))

        # Display the resulting frame
        cv2.imshow('Original frame', frame)

        # Write the frame to the video
        out.write(frame)

        # Convert the frame to grayscale:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the grayscale frame
        cv2.imshow('Grayscale frame', gray_frame)

        frame_index = frame_index - 1
        # print("next index to read: '{}'".format(frame_index))
 
        # Press q on keyboard to exit the program:
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Break the loop
    else:
        break
 
# Release everything:
capture.release()
out.release()
cv2.destroyAllWindows()

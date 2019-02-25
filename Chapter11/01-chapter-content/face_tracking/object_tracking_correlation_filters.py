"""
Tracking any object using dlib discriminative correlation filter tracker
"""

# Import required packages:
import cv2
import dlib


def draw_text_info():
    """Draw text information"""

    # We set the position to be used for drawing text and the menu info:
    menu_pos = (10, 20)
    menu_pos_2 = (10, 40)
    menu_pos_3 = (10, 60)
    info_1 = "Use left click of the mouse to select the object to track"
    info_2 = "Use '1' to start tracking, '2' to reset tracking and 'q' to exit"

    # Write text:
    cv2.putText(frame, info_1, menu_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    cv2.putText(frame, info_2, menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    if tracking_state:
        cv2.putText(frame, "tracking", menu_pos_3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "not tracking", menu_pos_3, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))


# Structure to hold the coordinates of the object to track:
points = []


# This is the mouse callback function:
def mouse_event_handler(event, x, y, flags, param):
    # references to the global points variable
    global points

    # If left button is click, add the top left coordinates of the object to be tracked:
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]

    # If left button is released, add the bottom right coordinates of the object to be tracked:
    elif event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))


# Create the video capture to read from the webcam:
capture = cv2.VideoCapture(0)

# Set window name:
window_name = "Object tracking using dlib correlation filter algorithm"

# Create the window:
cv2.namedWindow(window_name)

# We bind mouse events to the created window:
cv2.setMouseCallback(window_name, mouse_event_handler)

# First step is to initialize the correlation tracker.
tracker = dlib.correlation_tracker()

# This variable will hold if we are currently tracking the object:
tracking_state = False

while True:
    # Capture frame from webcam:
    ret, frame = capture.read()

    # We draw a basic instructions to the user:
    draw_text_info()

    # We set and draw the rectangle where the object will be tracked if it has the two points:
    if len(points) == 2:
        cv2.rectangle(frame, points[0], points[1], (0, 0, 255), 3)
        dlib_rectangle = dlib.rectangle(points[0][0], points[0][1], points[1][0], points[1][1])

    # If tracking, update tracking and get the position of the tracked object to be drawn:
    if tracking_state == True:
        # Update tracking
        tracker.update(frame)
        # Get the position of the tracked object:
        pos = tracker.get_position()
        # Draw the position:
        cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 3)

    # We capture the keyboard event:
    key = 0xFF & cv2.waitKey(1)

    # Press '1' to start tracking using the selected region:
    if key == ord("1"):
        if len(points) == 2:
            # Start tracking:
            tracker.start_track(frame, dlib_rectangle)
            tracking_state = True
            points = []

    # Press '2' to stop tracking. This will reset the points:
    if key == ord("2"):
        points = []
        tracking_state = False

    # To exit, press 'q':
    if key == ord('q'):
        break

    # Show the resulting image:
    cv2.imshow(window_name, frame)

# Release everything:
capture.release()
cv2.destroyAllWindows()

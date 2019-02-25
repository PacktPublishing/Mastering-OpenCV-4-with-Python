"""
Face tracking using dlib frontal face detector for initialization and dlib discriminative correlation filter tracker
"""

# Import required packages:
import cv2
import dlib


def draw_text_info():
    """Draw text information"""

    # We set the position to be used for drawing text and the menu info:
    menu_pos_1 = (10, 20)
    menu_pos_2 = (10, 40)

    # Write text:
    cv2.putText(frame, "Use '1' to re-initialize tracking", menu_pos_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    if tracking_face:
        cv2.putText(frame, "tracking the face", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
    else:
        cv2.putText(frame, "detecting a face to initialize tracking...", menu_pos_2, cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255))


# Create the video capture to read from the webcam:
capture = cv2.VideoCapture(0)

# Load frontal face detector from dlib:
detector = dlib.get_frontal_face_detector()

# We initialize the correlation tracker.
tracker = dlib.correlation_tracker()

# This variable will hold if we are currently tracking the face:
tracking_face = False

while True:
    # Capture frame from webcam:
    ret, frame = capture.read()

    # We draw basic info:
    draw_text_info()

    if tracking_face is False:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Try to detect a face to initialize the tracker:
        rects = detector(gray, 0)
        # Check if we can start tracking (if we detected a face):
        if len(rects) > 0:
            # Start tracking:
            tracker.start_track(frame, rects[0])
            tracking_face = True

    if tracking_face is True:
        # Update tracking and print the peak to side-lobe ratio (measures how confident the tracker is):
        print(tracker.update(frame))
        # Get the position of the tracked object:
        pos = tracker.get_position()
        # Draw the position:
        cv2.rectangle(frame, (int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom())), (0, 255, 0), 3)

    # We capture the keyboard event
    key = 0xFF & cv2.waitKey(1)

    # Press '1' to re-initialize tracking (it will detect the face again):
    if key == ord("1"):
        tracking_face = False

    # To exit, press 'q':
    if key == ord('q'):
        break

    # Show the resulting image:
    cv2.imshow("Face tracking using dlib frontal face detector and correlation filters for tracking", frame)

# Release everything:
capture.release()
cv2.destroyAllWindows()

"""
Snapchat-based augmented reality OpenCV moustache overlay
"""

# Import required packages:
import cv2

# Load cascade classifiers for face and nose detection:
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")

# Load moustache image. The parameter -1 reads also de alpha channel
# Open 'moustaches.sgv' to see more moustaches that can be used
# Therefore, the loaded image has four channels (Blue, Green, Red, Alpha):
img_moustache = cv2.imread('moustache.png', -1)

# Create the mask for the moustache:
img_moustache_mask = img_moustache[:, :, 3]
# cv2.imshow("img moustache mask", img_moustache_mask)

# You can use a test image to adjust the ROIS:
test_face = cv2.imread("face_test.png")

# Convert moustache image to BGR (eliminate alpha channel):
img_moustache = img_moustache[:, :, 0:3]

# Create VideoCapture object to get images from the webcam:
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame from the VideoCapture object:
    ret, frame = video_capture.read()

    # Just for debugging purposes and to adjust the ROIS:
    # frame = test_face.copy()

    # Convert frame to grayscale:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the function 'detectMultiScale'
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over each detected face:
    for (x, y, w, h) in faces:
        # Draw a rectangle to see the detected face (debugging purposes):
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

        # Create the ROIS based on the size of the detected face:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detects a nose inside the detected face:
        noses = nose_cascade.detectMultiScale(roi_gray)

        for (nx, ny, nw, nh) in noses:
            # Draw a rectangle to see the detected nose (debugging purposes):
            # cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 255), 2)

            # Calculate the coordinates where the moustache will be placed:
            x1 = int(nx - nw / 2)
            x2 = int(nx + nw / 2 + nw)
            y1 = int(ny + nh / 2 + nh / 8)
            y2 = int(ny + nh + nh / 4 + nh / 6)

            if x1 < 0 or x2 < 0 or x2 > w or y2 > h:
                continue

            # Draw a rectangle to see where the moustache will be placed (debugging purposes):
            # cv2.rectangle(roi_color, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Calculate the width and height of the image with the moustache:
            img_moustache_res_width = int(x2 - x1)
            img_moustache_res_height = int(y2 - y1)

            # Resize the mask to be equal to the region were the glasses will be placed:
            mask = cv2.resize(img_moustache_mask, (img_moustache_res_width, img_moustache_res_height))

            # Create the invert of the mask:
            mask_inv = cv2.bitwise_not(mask)

            # Resize img_glasses to the desired (and previously calculated) size:
            img = cv2.resize(img_moustache, (img_moustache_res_width, img_moustache_res_height))

            # Take ROI from the BGR image:
            roi = roi_color[y1:y2, x1:x2]

            # Create ROI background and ROI foreground:
            roi_bakground = cv2.bitwise_and(roi, roi, mask=mask_inv)
            roi_foreground = cv2.bitwise_and(img, img, mask=mask)

            # Show both roi_bakground and roi_foreground (debugging purposes):
            # cv2.imshow('roi_bakground', roi_bakground)
            # cv2.imshow('roi_foreground', roi_foreground)

            # Add roi_bakground and roi_foreground to create the result:
            res = cv2.add(roi_bakground, roi_foreground)

            # Set res into the color ROI:
            roi_color[y1:y2, x1:x2] = res

            break

    # Display the resulting frame:
    cv2.imshow('Snapchat-based OpenCV moustache overlay', frame)

    # Press any key to exit:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything:
video_capture.release()
cv2.destroyAllWindows()

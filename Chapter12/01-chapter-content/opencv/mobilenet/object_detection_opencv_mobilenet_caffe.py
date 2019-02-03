"""
Object detection using OpenCV CNN module using MobileNet-SSD and caffe pre-trained models

MobileNetSSD_deploy.caffemodel:
 https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc
MobileNetSSD_deploy.prototxt
 https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/daef68a6c2f5fbb8c88404266aa28180646d17e0/MobileNetSSD_deploy.prototxt
"""

# Import required packages:
import cv2
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Load the serialized caffe model from disk:
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")

# Load input image:
image = cv2.imread("object_detection_test_image.png")

# Prepare labels of the network (20 class labels + background):
class_names = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car',
               8: 'cat', 9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse', 14: 'motorbike',
               15: 'person', 16: 'pottedplant', 17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}

# Create the blob with a size of (300,300), mean subtraction values (127.5, 127.5, 127.5):
# and also a scalefactor of 0.007843:
blob = cv2.dnn.blobFromImage(image, 0.007843, (300, 300), (127.5, 127.5, 127.5))
print(blob.shape)

# Feed the input blob to the network, perform inference and ghe the output:
net.setInput(blob)
detections = net.forward()

# Get inference time:
t, _ = net.getPerfProfile()
print('Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))

# Size of frame resize (300x300)
dim = 300

# Process all detections:
for i in range(detections.shape[2]):
    # Get the confidence of the prediction:
    confidence = detections[0, 0, i, 2]

    # Filter predictions by confidence:
    if confidence > 0.1:
        # Get the class label:
        class_id = int(detections[0, 0, i, 1])

        # Get the coordinates of the object location:
        xLeftBottom = int(detections[0, 0, i, 3] * dim)
        yLeftBottom = int(detections[0, 0, i, 4] * dim)
        xRightTop = int(detections[0, 0, i, 5] * dim)
        yRightTop = int(detections[0, 0, i, 6] * dim)

        # Factor for scale to original size of frame
        heightFactor = image.shape[0] / dim
        widthFactor = image.shape[1] / dim

        # Scale object detection to frame
        xLeftBottom = int(widthFactor * xLeftBottom)
        yLeftBottom = int(heightFactor * yLeftBottom)
        xRightTop = int(widthFactor * xRightTop)
        yRightTop = int(heightFactor * yRightTop)

        # Draw rectangle:
        cv2.rectangle(image, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop), (0, 255, 0), 2)

        # Draw label and confidence:
        if class_id in class_names:
            label = class_names[class_id] + ": " + str(confidence)
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
            yLeftBottom = max(yLeftBottom, labelSize[1])
            cv2.rectangle(image, (xLeftBottom, yLeftBottom - labelSize[1]),
                          (xLeftBottom + labelSize[0], yLeftBottom + 0), (0, 255, 0), cv2.FILLED)
            cv2.putText(image, label, (xLeftBottom, yLeftBottom), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(14, 8))
plt.suptitle("Object detection using OpenCV CNN module and MobileNet-SSD", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Show the output image
show_img_with_matplotlib(image, "MobileNet-SSD for object detection", 1)

# Show the Figure:
plt.show()

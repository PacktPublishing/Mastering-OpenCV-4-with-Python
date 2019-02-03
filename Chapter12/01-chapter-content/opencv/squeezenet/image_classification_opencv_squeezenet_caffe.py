"""
Image classification using OpenCV CNN module using SqueezeNet and caffe pre-trained models

(SqueezeNet v1.1 has 2.4x less computation than v1.0, without sacrificing accuracy.)

deploy.prototxt:
 https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/deploy.prototxt
squeezenet_v1.1.caffemodel:
 https://github.com/DeepScale/SqueezeNet/blob/master/SqueezeNet_v1.1/squeezenet_v1.1.caffemodel
"""

# Import required packages:
import cv2
import numpy as np
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


# Load the names of the classes:
rows = open('synset_words.txt').read().strip().split('\n')
classes = [r[r.find(' ') + 1:].split(',')[0] for r in rows]

# Load the serialized caffe model from disk:
net = cv2.dnn.readNetFromCaffe("squeezenet_v1.1_deploy.prototxt", "squeezenet_v1.1.caffemodel")

# Load input image:
image = cv2.imread("church.jpg")

# Create the blob with a size of (227,227), mean subtraction values (104, 117, 123)
blob = cv2.dnn.blobFromImage(image, 1, (227, 227), (104, 117, 123))
print(blob.shape)

# Feed the input blob to the network, perform inference and ghe the output:
net.setInput(blob)
preds = net.forward()

# Get inference time:
t, _ = net.getPerfProfile()
print('Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency()))

# Get the 10 indexes with the highest probability (in descending order)
# This way, the index with the highest prob (top prediction) will be the first:
preds = preds.reshape((1, len(classes)))
indexes = np.argsort(preds[0])[::-1][:10]

# We draw on the image the class and probability associated with the top prediction:
text = "label: {}\nprobability: {:.2f}%".format(classes[indexes[0]], preds[0][indexes[0]] * 100)
print(text)
y0, dy = 30, 30
for i, line in enumerate(text.split('\n')):
    y = y0 + i * dy
    cv2.putText(image, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

# Print top 10 prediction:
for (index, idx) in enumerate(indexes):
    print("{}. label: {}, probability: {:.10}".format(index + 1, classes[idx], preds[0][idx]))

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 6))
plt.suptitle("Image classification with OpenCV using SqueezeNet (v1.1) and caffe pre-trained models", fontsize=14,
             fontweight='bold')
fig.patch.set_facecolor('silver')

# Show the output image:
show_img_with_matplotlib(image, "SqueezeNet (v1.1) and caffe pre-trained models", 1)

# Show the Figure:
plt.show()

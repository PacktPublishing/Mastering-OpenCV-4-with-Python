"""
Image classification in Keras using several models for image classification with weights trained on ImageNet
"""

# Import required packages:
import cv2
from keras.preprocessing import image
from keras.applications import inception_v3, vgg16, vgg19, resnet50, mobilenet, xception, nasnet, densenet
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
from matplotlib import pyplot as plt


def show_img_with_matplotlib(color_img, title, pos):
    """Shows an image using matplotlib capabilities"""

    # img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(color_img)
    plt.title(title)
    plt.axis('off')


def preprocessing_image(img_path, target_size, architecture):
    """Image preprocessing to be used for each Deep Learning architecture"""

    # Load image in PIL format
    img = image.load_img(img_path, target_size=target_size)
    # Convert PIL format to numpy array:
    x = image.img_to_array(img)
    # Convert the image/images into batch format:
    x = np.expand_dims(x, axis=0)
    # Pre-process (prepare) the image using the specific architecture:
    x = architecture.preprocess_input(x)
    return x


def put_text(img, model_name, decoded_preds, y_pos):
    """Show the predicted results in the image"""

    cv2.putText(img, "{}: {}, {:.2f}".format(model_name, decoded_preds[0][0][1], decoded_preds[0][0][2]),
                (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


# Path of the input image to be classified:
img_path = 'car.jpg'

# Load some available models:
model_inception_v3 = inception_v3.InceptionV3(weights='imagenet')
model_vgg_16 = vgg16.VGG16(weights='imagenet')
model_vgg_19 = vgg19.VGG19(weights='imagenet')
model_resnet_50 = resnet50.ResNet50(weights='imagenet')
model_mobilenet = mobilenet.MobileNet(weights='imagenet')
model_xception = xception.Xception(weights='imagenet')
model_nasnet_mobile = nasnet.NASNetMobile(weights='imagenet')
model_densenet_121 = densenet.DenseNet121(weights='imagenet')

# Prepare the image for the corresponding architecture:
x_inception_v3 = preprocessing_image(img_path, (299, 299), inception_v3)
x_vgg_16 = preprocessing_image(img_path, (224, 224), vgg16)
x_vgg_19 = preprocessing_image(img_path, (224, 224), vgg19)
x_resnet_50 = preprocessing_image(img_path, (224, 224), resnet50)
x_mobilenet = preprocessing_image(img_path, (224, 224), mobilenet)
x_xception = preprocessing_image(img_path, (299, 299), xception)
x_nasnet_mobile = preprocessing_image(img_path, (224, 224), nasnet)
x_densenet_121 = preprocessing_image(img_path, (224, 224), densenet)

# Get the predicted probabilities:
preds_inception_v3 = model_inception_v3.predict(x_inception_v3)
preds_vgg_16 = model_vgg_16.predict(x_vgg_16)
preds_vgg_19 = model_vgg_19.predict(x_vgg_19)
preds_resnet_50 = model_resnet_50.predict(x_resnet_50)
preds_mobilenet = model_mobilenet.predict(x_mobilenet)
preds_xception = model_xception.predict(x_xception)
preds_nasnet_mobile = model_nasnet_mobile.predict(x_nasnet_mobile)
preds_densenet_121 = model_nasnet_mobile.predict(x_densenet_121)

# Print the results (class, description, probability):
print('Predicted InceptionV3:', decode_predictions(preds_inception_v3, top=5)[0])
print('Predicted VGG16:', decode_predictions(preds_vgg_16, top=5)[0])
print('Predicted VGG19:', decode_predictions(preds_vgg_19, top=5)[0])
print('Predicted ResNet50:', decode_predictions(preds_resnet_50, top=5)[0])
print('Predicted MobileNet:', decode_predictions(preds_mobilenet, top=5)[0])
print('Predicted Xception:', decode_predictions(preds_xception, top=5)[0])
print('Predicted NASNetMobile:', decode_predictions(preds_nasnet_mobile, top=5)[0])
print('Predicted DenseNet121:', decode_predictions(preds_densenet_121, top=5)[0])

# Show results:
numpy_image = np.uint8(image.img_to_array(image.load_img(img_path))).copy()
numpy_image = cv2.resize(numpy_image, (500, 500))
numpy_image_res = numpy_image.copy()

put_text(numpy_image_res, "InceptionV3", decode_predictions(preds_inception_v3), 40)
put_text(numpy_image_res, "VGG16", decode_predictions(preds_vgg_16), 65)
put_text(numpy_image_res, "VGG19", decode_predictions(preds_vgg_19), 90)
put_text(numpy_image_res, "ResNet50", decode_predictions(preds_resnet_50), 115)
put_text(numpy_image_res, "MobileNet", decode_predictions(preds_mobilenet), 140)
put_text(numpy_image_res, "Xception", decode_predictions(preds_xception), 165)
put_text(numpy_image_res, "NASNetMobile", decode_predictions(preds_nasnet_mobile), 190)
put_text(numpy_image_res, "DenseNet121", decode_predictions(preds_densenet_121), 215)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(15, 7))
plt.suptitle("Image classification in Keras using several pre-trained models", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Show the output image:
show_img_with_matplotlib(numpy_image, "source image", 1)
show_img_with_matplotlib(numpy_image_res, "classification results", 2)

# Show the Figure:
plt.show()

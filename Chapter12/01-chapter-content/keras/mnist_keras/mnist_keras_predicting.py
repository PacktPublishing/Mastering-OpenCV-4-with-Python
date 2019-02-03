"""
Using a pre-trained Neural Network for predicting new handwritten digits in 'OpenCV images' using Keras
"""

import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import cv2
import numpy as np


def create_model():
    """Create the model using Sequential model"""

    # Create a sequential model (a simple NN is created) adding a softmax activation at the end with 10 units:
    model = Sequential()
    model.add(Dense(units=128, activation="relu", input_shape=(784,)))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=128, activation="relu"))
    model.add(Dense(units=10, activation="softmax"))

    # Compile the model using the loss function "categorical_crossentropy" and Stocastic Gradient Descent optimizer:
    model.compile(optimizer=SGD(0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    # Return the created model
    return model


# Note: Images should have black background:
def load_digit(image_name):
    """Loads a digit and pre-process in order to have the proper format"""

    gray = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    gray = cv2.resize(gray, (28, 28))
    gray = gray.reshape((1, 784))

    return gray


# Loaad MNIST data and show the dimensions of the loaded data:
(train_x, train_y), (test_x, test_y) = mnist.load_data()
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# Reshape to have proper size:
train_x = train_x.reshape(60000, 784)
test_x = test_x.reshape(10000, 784)
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)

# Create the model:
model = create_model()

# Load parameters of the model from the saved mode file:
model.load_weights("mnist-model.h5")

# Get the accuracy when testing:
accuracy = model.evaluate(x=test_x, y=test_y, batch_size=32)

# Show the accuracy:
print("Accuracy: ", accuracy[1])

# Load some test images:
test_digit_0 = load_digit("digit_0.png")
test_digit_1 = load_digit("digit_1.png")
test_digit_2 = load_digit("digit_2.png")
test_digit_3 = load_digit("digit_3.png")
imgs = np.array([test_digit_0, test_digit_1, test_digit_2, test_digit_3])
imgs = imgs.reshape(4, 784)

# Predict the class of the loaded images
prediction_class = model.predict_classes(imgs)

# Print the predicted classes:
print("Class: ", prediction_class)

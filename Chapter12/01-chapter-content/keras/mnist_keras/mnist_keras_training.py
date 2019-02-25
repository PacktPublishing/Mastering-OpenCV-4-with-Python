"""
Training Neural Network model using MNIST Dataset and saving the created model in Keras
"""

# Import required packages:
import keras
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD


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


# Load MNIST data and show the dimensions of the loaded data:
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

# Use the created model for training:
model.fit(train_x, train_y, batch_size=32, epochs=10, verbose=1)

# Save the created model:
model.save("mnist-model.h5")

# Get the accuracy when testing:
accuracy = model.evaluate(x=test_x, y=test_y, batch_size=32)

# Show the accuracy:
print("Accuracy: ", accuracy[1])

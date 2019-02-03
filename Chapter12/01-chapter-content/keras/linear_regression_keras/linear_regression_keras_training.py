"""
Training a linear regression model using Keras
"""

# Import required packages:
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Number of points:
N = 50

# Make random numbers predictable:
np.random.seed(101)

# Generate random data composed by 50 (N = 50) points:
x = np.linspace(0, N, N)
y = 3 * np.linspace(0, N, N) + np.random.uniform(-10, 10, N)


def get_weights(model):
    """Get weights of w and b"""

    w = model.get_weights()[0][0][0]
    b = model.get_weights()[1][0]
    return w, b


def create_model():
    """Create the model using Sequential model"""

    # Create a sequential model:
    model = Sequential()
    # All we need is a single connection so we use a Dense layer with linear activation:
    model.add(Dense(input_dim=1, units=1, activation="linear", kernel_initializer="uniform"))
    # Compile the model defining mean squared error(mse) as the loss
    model.compile(optimizer=Adam(lr=0.1), loss='mse')

    # Return the created model
    return model


# Get the created model:
linear_reg_model = create_model()

# Show weights at beginning (initialization values):
w_init, b_init = get_weights(linear_reg_model)
print('Linear regression model is initialized with weights w: {}, b: {}'.format(w_init, b_init))

# Feed the data using fit function:
linear_reg_model.fit(x, y, epochs=100, validation_split=0.2, verbose=1)

# Show weights when the training is done (learned parameters):
w_final, b_final = get_weights(linear_reg_model)
print('Linear regression model is trained with weights w: {}, b: {}'.format(w_final, b_final))

# Calculate the predictions:
predictions = w_final * x + b_final

# Saving weights:
linear_reg_model.save_weights("my_model.h5")

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(8, 5))
plt.suptitle("Linear regression using Keras", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot training data:
plt.subplot(1, 2, 1)
plt.plot(x, y, 'ro', label='Original data')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")

# Plot results:
plt.subplot(1, 2, 2)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, predictions, label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Result')
plt.legend()

# Show the Figure:
plt.show()

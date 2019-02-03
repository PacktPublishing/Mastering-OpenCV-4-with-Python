"""
Testing a linear regression model using Keras
"""

# Import required packages
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Number of points:
N = 50

# Number of points to predict:
M = 3

# Define 'M' more points to get the predictions using the trained model:
new_x = np.linspace(N + 1, N + 10, M)

# Make random numbers predictable:
np.random.seed(101)

# Generate random data composed by 50 (N = 50) points:
x = np.linspace(0, N, N)
y = 3 * np.linspace(0, N, N) + np.random.uniform(-10, 10, N)


def get_weights(model):
    m = model.get_weights()[0][0][0]
    b = model.get_weights()[1][0]
    return m, b


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


# Get the created model
linear_reg_model = create_model()

# Load weights:
linear_reg_model.load_weights('my_model.h5')

# Show weights when the training is done (learned parameters):
m_final, b_final = get_weights(linear_reg_model)
print('Linear regression model is trained with weights w: {}, b: {}'.format(m_final, b_final))

# Get the predictions of the training data:
predictions = linear_reg_model.predict(x)

# Get new predictions:
new_predictions = linear_reg_model.predict(new_x)

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 5))
plt.suptitle("Linear regression using Keras", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot training data:
plt.subplot(1, 3, 1)
plt.plot(x, y, 'ro', label='Original data')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")

# Plot results:
plt.subplot(1, 3, 2)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, predictions, label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Result')
plt.legend()

# Plot new predicted data:
plt.subplot(1, 3, 3)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, predictions, label='Fitted line')
plt.plot(new_x, new_predictions, 'bo', label='New predicted data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predicting new points')
plt.legend()

# Show the Figure:
plt.show()

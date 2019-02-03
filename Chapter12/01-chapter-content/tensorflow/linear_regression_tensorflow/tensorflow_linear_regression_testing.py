"""
Testing a linear regression model using TensorFlow
"""

# Import required packages:
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Number of points:
N = 50

# Make random numbers predictable:
np.random.seed(101)
tf.set_random_seed(101)

# Generate random data composed by 50 (N = 50) points:
x = np.linspace(0, N, N)
y = 3 * np.linspace(0, N, N) + np.random.uniform(-10, 10, N)

# Number of points to predict:
M = 3

# Define 'M' more points to get the predictions using the trained model:
new_x = np.linspace(N + 1, N + 10, M)

# Restore the model.
# First step when loading a model is to load the graph from '.meta':
tf.reset_default_graph()
imported_meta = tf.train.import_meta_graph("linear_regression.meta")

# The second step when loading a model is to load the values of the variables:
# Note that values only exist within a session
with tf.Session() as sess:
    imported_meta.restore(sess, './linear_regression')
    # Run the model to get the values of the variables W, b and new prediction values:
    W_estimated = sess.run('W:0')
    b_estimated = sess.run('b:0')
    new_predictions = sess.run(['y_model:0'], {'X:0': new_x})

# Reshape for proper visualization:
new_predictions = np.reshape(new_predictions, (M, -1))

# Calculate the predictions:
predictions = W_estimated * x + b_estimated

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(12, 5))
plt.suptitle("Linear regression using TensorFlow", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Plot training data:
plt.subplot(1, 3, 1)
plt.plot(x, y, 'ro', label='Original data')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")
plt.legend()

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
